// The speculative decoding toggle reads and sets one server-wide state through /speculative.

import { SpeculativeService } from '$lib/services/speculative.service';
import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
	vi.unstubAllGlobals();
});

function stubFetch(state: Record<string, unknown>) {
	const fetchMock = vi.fn(
		async () =>
			new Response(JSON.stringify(state), {
				headers: { 'Content-Type': 'application/json' },
				status: 200
			})
	);

	vi.stubGlobal('fetch', fetchMock);

	return fetchMock;
}

describe('speculative decoding service', () => {
	it('reads the server state', async () => {
		const fetchMock = stubFetch({ available: true, enabled: true, max_slots: 3 });
		const state = await SpeculativeService.fetch();

		expect(state).toEqual({ available: true, enabled: true, max_slots: 3 });
		expect(String((fetchMock.mock.calls[0] as unknown as [string])[0])).toContain('speculative');
	});

	it('posts the new setting', async () => {
		const fetchMock = stubFetch({ available: true, enabled: false, max_slots: 3 });
		const state = await SpeculativeService.set(false);
		const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];

		expect(state.enabled).toBe(false);
		expect(String(url)).toContain('speculative');
		expect(init.method).toBe('POST');
		expect(JSON.parse(init.body as string)).toEqual({ enabled: false });
	});
});
