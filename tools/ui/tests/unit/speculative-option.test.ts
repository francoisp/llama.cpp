// The "Speculative decoding" setting reaches the server as the `speculative` request field:
// on by default, and `false` once the user turns it off.

import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/services/database.service', () => ({
	DatabaseService: {}
}));

import { SETTING_CONFIG_DEFAULT, SETTINGS_KEYS } from '$lib/constants';
import { ChatService } from '$lib/services/chat.service';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';

afterEach(() => {
	settingsStore.config.speculative = SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.SPECULATIVE] as boolean;
	vi.unstubAllGlobals();
});

async function requestBody(options: { speculative?: boolean }): Promise<Record<string, unknown>> {
	const fetchMock = vi.fn(
		async () =>
			new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), {
				headers: { 'Content-Type': 'application/json' },
				status: 200
			})
	);

	vi.stubGlobal('fetch', fetchMock);

	await ChatService.sendMessage([{ content: 'hi', role: 'user' }], { ...options, stream: false });

	expect(fetchMock).toHaveBeenCalledTimes(1);
	const init = (fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1];

	return JSON.parse(init.body as string);
}

describe('speculative decoding setting', () => {
	it('is on by default', () => {
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.SPECULATIVE]).toBe(true);
		expect(chatStore.getApiOptions().speculative).toBe(true);
	});

	it('passes the user choice to the chat options', () => {
		settingsStore.config.speculative = false;

		expect(chatStore.getApiOptions().speculative).toBe(false);
	});

	it('sends the field in the request body only when set', async () => {
		expect((await requestBody({ speculative: false })).speculative).toBe(false);
		expect((await requestBody({ speculative: true })).speculative).toBe(true);
		expect('speculative' in (await requestBody({}))).toBe(false);
	});
});
