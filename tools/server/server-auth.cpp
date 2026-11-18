#include "server-auth.h"

#include "hash/hash.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>

#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef __linux__
#include <sys/random.h>
#endif

namespace fs = std::filesystem;

static const struct {
    uint32_t     perm;
    const char * name;
} k_perms[] = {
    { SERVER_PERM_CHAT,     "chat"     },
    { SERVER_PERM_STEERING, "steering" },
    { SERVER_PERM_CONTROL,  "control"  },
    { SERVER_PERM_TOOLS,    "tools"    },
    { SERVER_PERM_ADMIN,    "admin"    },
};

static constexpr size_t k_name_max = 80;

// key material from the kernel; std::random_device may use RDRAND, which some firmware breaks
static bool auth_random_bytes(uint8_t * buf, size_t n) {
#ifdef __linux__
    size_t got = 0;
    while (got < n) {
        const ssize_t r = getrandom(buf + got, n - got, 0);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        got += (size_t) r;
    }
    return true;
#else
    std::random_device rd;
    for (size_t i = 0; i < n; i++) {
        buf[i] = (uint8_t) (rd() & 0xff);
    }
    return true;
#endif
}

static std::string auth_trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace((unsigned char) s[b]))     { b++; }
    while (e > b && std::isspace((unsigned char) s[e - 1])) { e--; }
    return s.substr(b, e - b);
}

static std::string auth_hex(const std::array<uint8_t, 32> & d) {
    static const char * digits = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < 32; i++) {
        out[2*i]     = digits[d[i] >> 4];
        out[2*i + 1] = digits[d[i] & 15];
    }
    return out;
}

static bool auth_unhex(const std::string & s, std::array<uint8_t, 32> & d) {
    if (s.size() != 64) {
        return false;
    }
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') { return c - '0'; }
        if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
        if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
        return -1;
    };
    for (size_t i = 0; i < 32; i++) {
        const int hi = nib(s[2*i]);
        const int lo = nib(s[2*i + 1]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        d[i] = (uint8_t) (hi << 4 | lo);
    }
    return true;
}

// compares every byte, so the time taken does not tell how much of a guess was right
static bool auth_digest_equal(const std::array<uint8_t, 32> & a, const std::array<uint8_t, 32> & b) {
    uint8_t diff = 0;
    for (size_t i = 0; i < 32; i++) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

// a name fits on its line: no control characters, at most k_name_max bytes
static std::string auth_clean_name(const std::string & name) {
    std::string out;
    for (char c : name) {
        out += (unsigned char) c < 0x20 || c == 0x7f ? ' ' : c;
    }
    out = auth_trim(out);
    if (out.size() > k_name_max) {
        out.resize(k_name_max);
        // do not cut a UTF-8 sequence in half: drop the last character if it lost bytes
        size_t start = out.size() - 1;
        while (start > 0 && ((unsigned char) out[start] & 0xC0) == 0x80) {
            start--;
        }
        const unsigned char lead = out[start];
        const size_t len = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : 4;
        if (out.size() - start < len) {
            out.resize(start);
        }
    }
    return out;
}

static bool auth_parse_perms(const std::string & csv, uint32_t & out, std::string & bad) {
    out = SERVER_PERM_NONE;
    for (const auto & raw : string_split<std::string>(csv, ',')) {
        const std::string p = auth_trim(raw);
        if (p.empty()) {
            continue;
        }
        bool found = false;
        for (const auto & e : k_perms) {
            if (p == e.name) {
                out |= e.perm;
                found = true;
            }
        }
        if (!found) {
            bad = p;
            return false;
        }
    }
    return true;
}

static bool auth_perms_from_json(const json & v, uint32_t & out, std::string & err) {
    if (!v.is_array()) {
        err = "\"perms\" must be an array of permission names";
        return false;
    }
    std::string csv;
    for (const auto & p : v) {
        if (!p.is_string()) {
            err = "\"perms\" must be an array of permission names";
            return false;
        }
        csv += p.get<std::string>() + ",";
    }
    std::string bad;
    if (!auth_parse_perms(csv, out, bad)) {
        err = "unknown permission \"" + bad + "\"";
        return false;
    }
    return true;
}

// one line of the key file; false when the line grants nothing (a comment, a blank or a line it cannot read)
static bool auth_parse_line(const std::string & raw, server_api_key & key, std::string & err) {
    std::string s = auth_trim(raw);
    if (s.empty() || s[0] == '#') {
        return false;
    }

    // the name is a trailing comment
    for (size_t i = 1; i < s.size(); i++) {
        if (s[i] == '#' && std::isspace((unsigned char) s[i - 1])) {
            key.name = auth_clean_name(s.substr(i + 1));
            s = auth_trim(s.substr(0, i));
            break;
        }
    }

    std::istringstream in(s);
    std::vector<std::string> fields;
    for (std::string f; in >> f;) {
        fields.push_back(f);
    }

    key.stored = fields[0];
    if (string_starts_with(key.stored, "sha256:")) {
        if (!auth_unhex(key.stored.substr(7), key.digest)) {
            err = "sha256: must be followed by 64 hex digits";
            return false;
        }
        key.hashed = true;
    } else {
        key.digest = hash_sha256(key.stored.data(), key.stored.size());
        if (key.stored.size() >= 16) {
            key.hint = key.stored.substr(key.stored.size() - 4);
        }
    }
    key.id    = auth_hex(key.digest).substr(0, 12);
    key.perms = SERVER_PERM_DEFAULT;

    for (size_t i = 1; i < fields.size(); i++) {
        const std::string & f = fields[i];
        if (string_starts_with(f, "perms=")) {
            std::string bad;
            if (!auth_parse_perms(f.substr(6), key.perms, bad)) {
                err = "unknown permission \"" + bad + "\"";
                return false;
            }
        } else if (string_starts_with(f, "hint=") && key.hashed) {
            key.hint = f.substr(5, 8);
        } else if (f == "disabled") {
            key.disabled = true;
        } else {
            // a misspelt field must not leave the key with more than was meant
            err = "unknown field \"" + f.substr(0, f.find('=')) + "\"";
            return false;
        }
    }
    return true;
}

static std::string auth_format_line(const server_api_key & key) {
    std::string line = key.stored;
    line += " perms=" + string_join(server_auth::perm_names(key.perms), ",");
    if (key.hashed && !key.hint.empty()) {
        line += " hint=" + key.hint;
    }
    if (key.disabled) {
        line += " disabled";
    }
    if (!key.name.empty()) {
        line += "  # " + key.name;
    }
    return line;
}

json server_api_key::to_json() const {
    return json {
        {"id",       id},
        {"name",     name},
        {"hint",     hint},
        {"perms",    server_auth::perm_names(perms)},
        {"disabled", disabled},
        {"hashed",   hashed},
        {"source",   cli ? "cli" : "file"},
    };
}

const char * server_auth::perm_name(uint32_t perm) {
    for (const auto & e : k_perms) {
        if (e.perm == perm) {
            return e.name;
        }
    }
    return "unknown";
}

std::vector<std::string> server_auth::perm_names(uint32_t perms) {
    std::vector<std::string> out;
    for (const auto & e : k_perms) {
        if (perms & e.perm) {
            out.push_back(e.name);
        }
    }
    return out;
}

std::string server_auth::secret_from_headers(const std::string & authorization, const std::string & x_api_key) {
    std::string secret = authorization.empty() ? x_api_key : authorization;
    static const std::string bearer = "Bearer ";
    if (secret.compare(0, bearer.size(), bearer) == 0) {
        secret = secret.substr(bearer.size());
    }
    return auth_trim(secret);
}

uint32_t server_auth::required_perm(const std::string & method, const std::string & path) {
    const bool is_post = method == "POST";

    if (path == "/keys/me") {
        return SERVER_PERM_NONE;
    }
    if (path == "/keys" || string_starts_with(path, "/keys/")) {
        return SERVER_PERM_ADMIN;
    }
    if (path == "/steering") {
        return is_post ? SERVER_PERM_STEERING : SERVER_PERM_CHAT;
    }
    if (path == "/speculative" || path == "/props" || path == "/lora-adapters") {
        return is_post ? SERVER_PERM_CONTROL : SERVER_PERM_CHAT;
    }
    // slots show every user's prompt, and saving or restoring one replaces a context
    if (path == "/slots" || string_starts_with(path, "/slots/")) {
        return SERVER_PERM_CONTROL;
    }
    if (path == "/models/load" || path == "/models/unload" || (path == "/models" && is_post)) {
        return SERVER_PERM_CONTROL;
    }
    if (path == "/tools" || path == "/cors-proxy") {
        return SERVER_PERM_TOOLS;
    }
    return SERVER_PERM_CHAT;
}

bool server_auth::init(const std::vector<std::string> & keys, const std::string & key_file, std::string & err) {
    for (const auto & k : keys) {
        server_api_key key;
        key.stored = k;
        key.digest = hash_sha256(k.data(), k.size());
        key.id     = auth_hex(key.digest).substr(0, 12);
        key.hint   = k.size() >= 16 ? k.substr(k.size() - 4) : "";
        key.name   = "--api-key";
        key.perms  = SERVER_PERM_DEFAULT;
        key.cli    = true;
        cli_keys.push_back(key);
    }
    path     = key_file;
    enabled_ = !cli_keys.empty() || !path.empty();

    if (!path.empty()) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!load(err)) {
            return false;
        }
#ifndef _WIN32
        std::error_code ec;
        const auto p = fs::status(path, ec).permissions();
        if (!ec && (p & (fs::perms::group_all | fs::perms::others_all)) != fs::perms::none) {
            SRV_WRN("api keys: %s is readable by other users, run chmod 600 on it\n", path.c_str());
        }
#endif
    }
    if (cli_keys.size() == 1) {
        SRV_INF("api keys: 1 key from --api-key (****%s)\n", cli_keys[0].hint.c_str());
    } else if (cli_keys.size() > 1) {
        SRV_INF("api keys: %zu keys from --api-key\n", cli_keys.size());
    }
    return true;
}

server_auth::file_sig server_auth::read_sig() const {
    file_sig s;
    std::error_code ec;
    s.mtime = fs::last_write_time(path, ec);
    if (ec) {
        return s;
    }
    s.size = fs::file_size(path, ec);
    if (ec) {
        return s;
    }
    s.exists = true;
    return s;
}

bool server_auth::load(std::string & err) {
    const file_sig s = read_sig();
    std::ifstream f(path, std::ios::binary);
    if (!s.exists || !f) {
        err = "cannot read API key file " + path;
        return false;
    }

    std::vector<std::string>    new_lines;
    std::vector<server_api_key> new_keys;
    size_t n_bad   = 0;
    size_t n_admin = 0;
    for (std::string raw; std::getline(f, raw);) {
        if (!raw.empty() && raw.back() == '\r') {
            raw.pop_back();
        }
        server_api_key key;
        std::string    line_err;
        if (auth_parse_line(raw, key, line_err)) {
            const bool dup = std::any_of(new_keys.begin(), new_keys.end(), [&](const server_api_key & k) {
                return auth_digest_equal(k.digest, key.digest);
            });
            if (dup) {
                SRV_WRN("api keys: %s:%zu repeats key %s, only its first line counts\n", path.c_str(), new_lines.size() + 1, key.id.c_str());
            } else {
                key.line = (int) new_lines.size();
                n_admin += !key.disabled && (key.perms & SERVER_PERM_ADMIN) ? 1 : 0;
                new_keys.push_back(std::move(key));
            }
        } else if (!line_err.empty()) {
            // never print the line: it may hold a key
            SRV_WRN("api keys: %s:%zu grants nothing: %s\n", path.c_str(), new_lines.size() + 1, line_err.c_str());
            n_bad++;
        }
        new_lines.push_back(raw);
    }

    lines       = std::move(new_lines);
    file_keys   = std::move(new_keys);
    sig         = s;
    load_failed = false;
    SRV_INF("api keys: %zu keys from %s (%zu with admin)%s\n", file_keys.size(), path.c_str(), n_admin,
            n_bad ? string_format(", %zu lines ignored", n_bad).c_str() : "");
    return true;
}

void server_auth::reload_if_changed() {
    if (path.empty()) {
        return;
    }
    const file_sig s = read_sig();
    if (s == sig && !load_failed) {
        return;
    }
    std::string err;
    if (!load(err)) {
        // fail closed: a key file that is gone or unreadable grants nothing until it is back
        if (!load_failed) {
            SRV_WRN("api keys: %s, no key from it is accepted until it can be read\n", err.c_str());
        }
        file_keys.clear();
        lines.clear();
        sig         = s;
        load_failed = true;
    }
}

std::optional<server_api_key> server_auth::find(const std::string & secret) {
    if (secret.empty()) {
        return std::nullopt;
    }
    const auto digest = hash_sha256(secret.data(), secret.size());

    std::lock_guard<std::mutex> lock(mtx);
    reload_if_changed();

    std::optional<server_api_key> found;
    for (const auto * list : { &cli_keys, &file_keys }) {
        for (const auto & k : *list) {
            if (auth_digest_equal(k.digest, digest) && !found) {
                found = k;
            }
        }
    }
    if (found && found->disabled) {
        return std::nullopt;
    }
    return found;
}

server_api_key * server_auth::find_by_id(const std::string & id) {
    for (auto & k : file_keys) {
        if (k.id == id) {
            return &k;
        }
    }
    return nullptr;
}

bool server_auth::write(const std::vector<std::string> & new_lines, std::string & err) {
    std::error_code ec;
    fs::path target = fs::weakly_canonical(path, ec); // write through a symlink, not over it
    if (ec) {
        target = path;
    }

    // a new file next to the old one, renamed over it: a reader sees the old file or the new one, never half of one
    uint8_t r[4] = {};
    auth_random_bytes(r, sizeof(r));
    const fs::path tmp = target.parent_path() / string_format(".%s.tmp%02x%02x%02x%02x", target.filename().string().c_str(), r[0], r[1], r[2], r[3]);

    FILE * f = fopen(tmp.string().c_str(), "wbx");
    if (!f) {
        err = "cannot write " + tmp.string() + ": " + strerror(errno);
        return false;
    }
    fs::permissions(tmp, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace, ec);

    bool ok = true;
    for (const auto & l : new_lines) {
        ok = ok && fwrite(l.data(), 1, l.size(), f) == l.size() && fputc('\n', f) != EOF;
    }
    ok = ok && fflush(f) == 0;
#ifndef _WIN32
    ok = ok && fsync(fileno(f)) == 0;
#endif
    ok = fclose(f) == 0 && ok;

    if (ok) {
        fs::rename(tmp, target, ec);
        ok = !ec;
    }
    if (!ok) {
        err = "cannot write " + target.string() + (ec ? ": " + ec.message() : "");
        fs::remove(tmp, ec);
        return false;
    }

    std::string load_err;
    if (!load(load_err)) {
        err = load_err;
        return false;
    }
    return true;
}

json server_auth::me(const std::optional<server_api_key> & caller) const {
    if (!enabled_) {
        return json {
            {"auth",     false},
            {"perms",    perm_names(SERVER_PERM_DEFAULT)},
            {"admin",    false},
            {"key_file", false},
        };
    }
    json res = caller ? caller->to_json() : json::object();
    res["auth"]     = true;
    res["admin"]    = caller && (caller->perms & SERVER_PERM_ADMIN);
    res["key_file"] = !path.empty();
    return res;
}

server_auth_result server_auth::list(const server_api_key & caller) {
    std::lock_guard<std::mutex> lock(mtx);
    reload_if_changed();

    json keys = json::array();
    for (const auto * l : { &cli_keys, &file_keys }) {
        for (const auto & k : *l) {
            json j = k.to_json();
            j["self"] = k.id == caller.id;
            keys.push_back(j);
        }
    }
    json perms = json::array();
    for (const auto & e : k_perms) {
        perms.push_back(e.name);
    }
    return { 200, json {
        {"key_file", path.empty() ? json(nullptr) : json(path)},
        {"keys",     keys},
        {"perms",    perms},
    }};
}

static server_auth_result auth_error(int status, const std::string & message) {
    const char * type = status == 404 ? "not_found_error" : status == 409 ? "conflict_error" : status >= 500 ? "server_error" : "invalid_request_error";
    return { status, json {{"error", {{"code", status}, {"message", message}, {"type", type}}}} };
}

server_auth_result server_auth::create(const server_api_key & caller, const json & body) {
    GGML_UNUSED(caller);
    if (path.empty()) {
        return auth_error(409, "keys can only be added to a key file: start the server with --api-key-file");
    }

    std::string err;
    uint32_t perms = SERVER_PERM_CHAT;
    if (body.contains("perms") && !auth_perms_from_json(body.at("perms"), perms, err)) {
        return auth_error(400, err);
    }
    const std::string name = auth_clean_name(json_value(body, "name", std::string()));

    // 256 random bits; only the hash goes in the file, so the key is shown once
    std::array<uint8_t, 32> bytes;
    if (!auth_random_bytes(bytes.data(), bytes.size())) {
        return auth_error(500, "cannot get random bytes for a new key");
    }
    const std::string secret = "llk-" + auth_hex(bytes);

    server_api_key key;
    key.digest = hash_sha256(secret.data(), secret.size());
    key.stored = "sha256:" + auth_hex(key.digest);
    key.id     = auth_hex(key.digest).substr(0, 12);
    key.hint   = secret.substr(secret.size() - 4);
    key.hashed = true;
    key.name   = name;
    key.perms  = perms;

    std::lock_guard<std::mutex> lock(mtx);
    reload_if_changed();
    if (load_failed) {
        return auth_error(500, "the key file cannot be read, fix it before adding keys");
    }

    std::vector<std::string> new_lines = lines;
    new_lines.push_back(auth_format_line(key));
    if (!write(new_lines, err)) {
        return auth_error(500, err);
    }
    SRV_INF("api keys: added %s (%s) with %s\n", key.id.c_str(), key.name.c_str(), string_join(perm_names(key.perms), ",").c_str());

    json res = key.to_json();
    res["key"] = secret;
    return { 200, res };
}

server_auth_result server_auth::update(const server_api_key & caller, const std::string & id, const json & body) {
    std::lock_guard<std::mutex> lock(mtx);
    reload_if_changed();

    server_api_key * found = find_by_id(id);
    if (!found) {
        const bool cli = std::any_of(cli_keys.begin(), cli_keys.end(), [&](const server_api_key & k) { return k.id == id; });
        return cli ? auth_error(409, "this key comes from --api-key and changes only with a restart")
                   : auth_error(404, "no key with id " + id);
    }

    server_api_key key = *found;
    std::string err;
    if (body.contains("name")) {
        key.name = auth_clean_name(json_value(body, "name", std::string()));
    }
    if (body.contains("perms") && !auth_perms_from_json(body.at("perms"), key.perms, err)) {
        return auth_error(400, err);
    }
    if (body.contains("disabled")) {
        key.disabled = json_value(body, "disabled", false);
    }

    // the key in use cannot lock its holder out
    if (key.id == caller.id && (key.disabled || !(key.perms & SERVER_PERM_ADMIN))) {
        return auth_error(409, "you cannot disable or take admin from the key you are signed in with");
    }

    std::vector<std::string> new_lines = lines;
    new_lines[key.line] = auth_format_line(key);
    if (!write(new_lines, err)) {
        return auth_error(500, err);
    }
    SRV_INF("api keys: changed %s (%s): %s%s\n", key.id.c_str(), key.name.c_str(),
            string_join(perm_names(key.perms), ",").c_str(), key.disabled ? ", disabled" : "");

    const server_api_key * now = find_by_id(id);
    return { 200, now ? now->to_json() : key.to_json() };
}

server_auth_result server_auth::remove(const server_api_key & caller, const std::string & id) {
    std::lock_guard<std::mutex> lock(mtx);
    reload_if_changed();

    server_api_key * found = find_by_id(id);
    if (!found) {
        const bool cli = std::any_of(cli_keys.begin(), cli_keys.end(), [&](const server_api_key & k) { return k.id == id; });
        return cli ? auth_error(409, "this key comes from --api-key and goes away only with a restart")
                   : auth_error(404, "no key with id " + id);
    }
    if (found->id == caller.id) {
        return auth_error(409, "you cannot delete the key you are signed in with");
    }

    const std::string name = found->name;
    std::vector<std::string> new_lines = lines;
    new_lines.erase(new_lines.begin() + found->line);
    std::string err;
    if (!write(new_lines, err)) {
        return auth_error(500, err);
    }
    SRV_INF("api keys: deleted %s (%s)\n", id.c_str(), name.c_str());
    return { 200, json {{"deleted", id}} };
}
