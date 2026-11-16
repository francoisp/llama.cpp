<script lang="ts">
	import { Check, Compass } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { SteeringService } from '$lib/services/steering.service';
	import type { ApiSteeringState, ApiSteeringVector } from '$lib/types';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		disabled?: boolean;
	}

	let { disabled = false }: Props = $props();

	let steering = $state<ApiSteeringState | null>(null);
	let isBusy = $state(false);

	let vectors = $derived(steering?.available ?? []);
	let active = $derived(vectors.find((vector) => vector.file === steering?.file));
	let activeLabel = $derived(active?.label ?? steering?.file ?? null);

	onMount(async () => {
		try {
			steering = await SteeringService.fetch();
		} catch {
			// a server without steering: the dropdown stays hidden
			steering = null;
		}
	});

	async function select(file: string | null): Promise<void> {
		if (file === (steering?.file ?? null)) {
			return;
		}

		isBusy = true;
		try {
			steering = await SteeringService.select(file);
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to change steering');
		} finally {
			isBusy = false;
		}
	}

	function scales(vector: ApiSteeringVector): string {
		return [
			vector.attn ? `attn ${vector.attn}` : '',
			vector.ffn ? `ffn ${vector.ffn}` : '',
			vector.resid ? `resid ${vector.resid}` : ''
		]
			.filter(Boolean)
			.join(' · ');
	}
</script>

{#if vectors.length > 0}
	<DropdownMenu.Root>
		<DropdownMenu.Trigger disabled={disabled || isBusy}>
			{#snippet child({ props })}
				<Button
					{...props}
					aria-label="Directional steering"
					class="-mr-px h-6 min-w-6 gap-1 rounded-sm bg-transparent backdrop-blur-none! {activeLabel
						? 'px-1.75!'
						: 'px-1'} py-0 text-xs hover:bg-muted-foreground/15 data-[state=open]:bg-muted-foreground/15"
					variant="ghost"
				>
					<span class="flex items-center gap-0.75 {activeLabel ? '' : 'text-muted-foreground'}">
						<Compass class="size-3.25 max-md:size-4 shrink-0" />

						{#if activeLabel}
							<span>{activeLabel}</span>
						{/if}
					</span>
				</Button>
			{/snippet}
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="end" class="min-w-56">
			<DropdownMenu.Item class="gap-3" onSelect={() => select(null)}>
				{#if !steering?.file}
					<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
				{:else}
					<div class="{ICON_CLASS_DEFAULT} shrink-0"></div>
				{/if}

				<span class="min-w-0 flex-1 truncate">Off</span>
			</DropdownMenu.Item>

			{#each vectors as vector (vector.file)}
				<DropdownMenu.Item class="gap-3" onSelect={() => select(vector.file)}>
					{#if vector.file === steering?.file}
						<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
					{:else}
						<div class="{ICON_CLASS_DEFAULT} shrink-0"></div>
					{/if}

					<span class="min-w-0 flex-1 truncate">{vector.label}</span>

					<span class="shrink-0 text-[11px] text-muted-foreground opacity-60">{scales(vector)}</span
					>
				</DropdownMenu.Item>
			{/each}

			<DropdownMenu.Separator />

			<p class="px-2 py-1 text-[11px] text-muted-foreground">
				Applies to new tokens, on every open chat
			</p>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
{/if}
