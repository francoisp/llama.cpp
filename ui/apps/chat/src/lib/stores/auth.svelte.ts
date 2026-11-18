/**
 * authStore - What the API key in the settings may do on this server
 *
 * Owns the answer of `/keys/me`, refreshed whenever the key in the settings changes. The server checks every
 * request on its own; this store only lets controls the key cannot use show as unavailable instead of failing
 * when clicked.
 */

import { ApiKeysService } from '$lib/services/api-keys.service';
import type { ApiKeyPermission, ApiKeysMe } from '$lib/types';

class AuthStore {
	me = $state<ApiKeysMe | null>(null);
	private requestId = 0;

	/** The key may manage the server's keys */
	get isAdmin(): boolean {
		return this.me?.admin === true;
	}

	/**
	 * Whether the key has a permission. Until the server answers, or when it cannot, everything counts as
	 * allowed: the server still refuses what the key may not do.
	 */
	can(perm: ApiKeyPermission): boolean {
		if (!this.me) return true;

		return this.me.admin || this.me.perms.includes(perm);
	}

	async refresh(): Promise<void> {
		const id = ++this.requestId;

		try {
			const me = await ApiKeysService.me();

			// a newer refresh, for a newer key, wins
			if (id === this.requestId) this.me = me;
		} catch {
			if (id === this.requestId) this.me = null;
		}
	}
}

export const authStore = new AuthStore();
