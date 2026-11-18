#pragma once

#include "server-common.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// API keys with permissions, read from --api-key and --api-key-file
//
// The key file holds one key per line, picked up again whenever the file changes, so a key can be added,
// changed or removed without a restart:
//
//   KEY [perms=chat,steering,control,tools,admin] [disabled]  # name
//   sha256:HEX [perms=...] [hint=XXXX] [disabled]              # name
//
// A line holds either the key itself or the SHA-256 of it; the server stores keys it creates as hashes, so the
// file never holds them. A key without perms= has every permission except admin, as every key had before
// permissions existed. Lines starting with # are comments, and a line the server cannot read grants nothing.

enum server_perm : uint32_t {
    SERVER_PERM_NONE     = 0,
    SERVER_PERM_CHAT     = 1u << 0, // inference, and reading server state
    SERVER_PERM_STEERING = 1u << 1, // changing the directional steering, which applies to every slot
    SERVER_PERM_CONTROL  = 1u << 2, // other server-wide settings: drafting, props, LoRA, slots, router models
    SERVER_PERM_TOOLS    = 1u << 3, // server tools and the MCP proxy
    SERVER_PERM_ADMIN    = 1u << 4, // managing the keys; implies every other permission
};

constexpr uint32_t SERVER_PERM_DEFAULT = SERVER_PERM_CHAT | SERVER_PERM_STEERING | SERVER_PERM_CONTROL | SERVER_PERM_TOOLS;
constexpr uint32_t SERVER_PERM_ALL     = SERVER_PERM_DEFAULT | SERVER_PERM_ADMIN;

struct server_api_key {
    std::string id;     // the first 12 hex digits of the key's SHA-256, stable across reloads
    std::string name;
    std::string hint;   // the last 4 characters of the key, when known
    std::array<uint8_t, 32> digest = {};
    uint32_t perms    = SERVER_PERM_NONE;
    bool     disabled = false;
    bool     hashed   = false; // the file holds the SHA-256, not the key
    bool     cli      = false; // from --api-key: cannot change without a restart
    int      line     = -1;    // index of its line in the key file

    // the key's first field as written in the file; never sent to a client
    std::string stored;

    json to_json() const;
};

struct server_auth_result {
    int  status = 200;
    json body;
};

class server_auth {
public:
    // false when the key file cannot be read at startup
    bool init(const std::vector<std::string> & cli_keys, const std::string & key_file, std::string & err);

    // whether requests need a key: true when keys were configured at startup, even if the file is empty now
    bool enabled() const { return enabled_; }

    // the enabled key matching a secret, after picking up changes to the key file
    std::optional<server_api_key> find(const std::string & secret);

    // the key a request presents, from "Authorization: Bearer KEY" or "X-Api-Key: KEY"
    static std::string secret_from_headers(const std::string & authorization, const std::string & x_api_key);

    // the permission a request needs, from its method and its path without the API prefix
    static uint32_t required_perm(const std::string & method, const std::string & path);

    static bool allows(uint32_t perms, uint32_t needed) {
        return (perms & SERVER_PERM_ADMIN) || (perms & needed) == needed;
    }

    static std::vector<std::string> perm_names(uint32_t perms);
    static const char *             perm_name(uint32_t perm);

    // /keys/me: what the caller may do
    json me(const std::optional<server_api_key> & caller) const;

    // admin operations: list, create, update and delete keys, rewriting the key file
    server_auth_result list(const server_api_key & caller);
    server_auth_result create(const server_api_key & caller, const json & body);
    server_auth_result update(const server_api_key & caller, const std::string & id, const json & body);
    server_auth_result remove(const server_api_key & caller, const std::string & id);

private:
    struct file_sig {
        bool exists = false;
        std::filesystem::file_time_type mtime;
        uintmax_t size = 0;
        bool operator==(const file_sig & o) const {
            return exists == o.exists && (!exists || (mtime == o.mtime && size == o.size));
        }
    };

    bool        enabled_ = false;
    std::string path;

    std::mutex                  mtx;
    std::vector<server_api_key> cli_keys;
    std::vector<server_api_key> file_keys;
    std::vector<std::string>    lines;      // the key file as last read, line by line
    file_sig                    sig;
    bool                        load_failed = false;

    file_sig read_sig() const;
    void     reload_if_changed();               // with mtx held
    bool     load(std::string & err);           // with mtx held
    bool     write(const std::vector<std::string> & new_lines, std::string & err); // with mtx held
    server_api_key * find_by_id(const std::string & id); // with mtx held
};
