/**
 * SteeringService - Reads and sets directional steering through `/steering`
 *
 * No reactive state. The server applies a change to every slot from the next
 * decode, and tokens already in the context keep the steering they had.
 */

import type { ApiSteeringState } from '$lib/types';
import { apiFetch, apiPost } from '$lib/utils';

export class SteeringService {
	/**
	 * Fetches the current steering and the vectors the server offers.
	 *
	 * @throws {Error} If the request fails, as it does on a server without steering
	 */
	static async fetch(): Promise<ApiSteeringState> {
		return apiFetch<ApiSteeringState>('./steering', { authOnly: true });
	}

	/**
	 * Loads a vector with the scales the server keeps for it, or unloads it.
	 *
	 * @param file - A file name from `available`, or null to unload
	 */
	static async select(file: string | null): Promise<ApiSteeringState> {
		return apiPost<ApiSteeringState>('./steering', { file });
	}
}
