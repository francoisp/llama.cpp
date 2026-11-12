<script lang="ts">
	import { Zap, ZapOff } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { SpeculativeService } from '$lib/services/speculative.service';
	import type { ApiSpeculativeState } from '$lib/types';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		disabled?: boolean;
	}

	let { disabled = false }: Props = $props();

	let speculative = $state<ApiSpeculativeState | null>(null);
	let isBusy = $state(false);

	let enabled = $derived(speculative?.enabled ?? false);

	onMount(async () => {
		try {
			speculative = await SpeculativeService.fetch();
		} catch {
			// a server without the endpoint: the toggle stays hidden
			speculative = null;
		}
	});

	async function toggle(): Promise<void> {
		if (!speculative) {
			return;
		}

		isBusy = true;
		try {
			speculative = await SpeculativeService.set(!speculative.enabled);
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to change speculative decoding');
		} finally {
			isBusy = false;
		}
	}
</script>

{#if speculative?.available}
	<Button
		aria-label="Speculative decoding"
		aria-pressed={enabled}
		class="-mr-px h-6 min-w-6 rounded-sm bg-transparent px-1 py-0 backdrop-blur-none! hover:bg-muted-foreground/15"
		disabled={disabled || isBusy}
		onclick={toggle}
		title={enabled
			? 'Speculative decoding is on for the whole server (new requests). Click to turn it off.'
			: 'Speculative decoding is off for the whole server (new requests). Click to turn it on.'}
		variant="ghost"
	>
		{#if enabled}
			<Zap class="size-3.25 shrink-0 max-md:size-4" />
		{:else}
			<ZapOff class="size-3.25 shrink-0 text-muted-foreground max-md:size-4" />
		{/if}
	</Button>
{/if}
