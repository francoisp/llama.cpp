/**
 * SpeculativeService - Reads and sets speculative decoding for the whole server through `/speculative`
 *
 * No reactive state. A change applies to the requests that start afterwards, on every slot; requests
 * that are already generating keep the setting they started with.
 */

import type { ApiSpeculativeState } from '$lib/types';
import { apiFetch, apiPost } from '$lib/utils';

export class SpeculativeService {
	/**
	 * Fetches whether speculative decoding is on and whether the server has a draft model.
	 *
	 * @throws {Error} If the request fails, as it does on a server without the endpoint
	 */
	static async fetch(): Promise<ApiSpeculativeState> {
		return apiFetch<ApiSpeculativeState>('./speculative', { authOnly: true });
	}

	/**
	 * Turns speculative decoding on or off for the whole server.
	 */
	static async set(enabled: boolean): Promise<ApiSpeculativeState> {
		return apiPost<ApiSpeculativeState>('./speculative', { enabled });
	}
}
