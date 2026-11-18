/**
 * ApiKeysService - Reads what the signed-in API key may do, and manages the server's keys through `/keys`
 *
 * No reactive state. Every call but `me` needs a key with the admin permission. The server writes changes to
 * its key file and applies them to the next request, without a restart.
 */

import type {
	ApiKeyCreated,
	ApiKeyEntry,
	ApiKeyPermission,
	ApiKeysList,
	ApiKeysMe
} from '$lib/types';
import { apiDelete, apiFetch, apiPost } from '$lib/utils';

export class ApiKeysService {
	/**
	 * Creates a key. The response holds the key itself, which the server does not keep.
	 */
	static async create(name: string, perms: ApiKeyPermission[]): Promise<ApiKeyCreated> {
		return apiPost<ApiKeyCreated>('./keys', { name, perms });
	}

	/**
	 * Lists the keys, the file they come from and the permissions a key can have.
	 */
	static async list(): Promise<ApiKeysList> {
		return apiFetch<ApiKeysList>('./keys', { authOnly: true });
	}

	/**
	 * Fetches the permissions of the key in the settings; on a server without keys, every one but admin.
	 */
	static async me(): Promise<ApiKeysMe> {
		return apiFetch<ApiKeysMe>('./keys/me', { authOnly: true });
	}

	/**
	 * Deletes a key; requests that use it fail from then on.
	 */
	static async remove(id: string): Promise<{ deleted: string }> {
		return apiDelete<{ deleted: string }>(`./keys/${encodeURIComponent(id)}`);
	}

	/**
	 * Renames a key, changes its permissions or turns it on or off.
	 */
	static async update(
		id: string,
		patch: Partial<Pick<ApiKeyEntry, 'disabled' | 'name' | 'perms'>>
	): Promise<ApiKeyEntry> {
		return apiPost<ApiKeyEntry>(`./keys/${encodeURIComponent(id)}`, patch);
	}
}
