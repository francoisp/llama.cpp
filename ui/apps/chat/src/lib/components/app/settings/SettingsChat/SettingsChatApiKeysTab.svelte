<script lang="ts">
	import { Copy, KeyRound, Plus, RefreshCw, Trash2 } from '@lucide/svelte';
	import { DialogConfirmation } from '$lib/components/app';
	import SettingsGroup from '$lib/components/app/settings/SettingsGroup.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import { Switch } from '$lib/components/ui/switch';
	import { ApiKeysService } from '$lib/services/api-keys.service';
	import { authStore } from '$lib/stores';
	import type { ApiKeyCreated, ApiKeyEntry, ApiKeyPermission, ApiKeysList } from '$lib/types';
	import { copyToClipboard } from '$lib/utils';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	const PERM_LABELS: Record<ApiKeyPermission, { label: string; help: string }> = {
		admin: { help: 'Manage these keys; includes every other permission', label: 'Admin' },
		chat: { help: 'Chat, completions, embeddings, and reading server state', label: 'Chat' },
		control: {
			help: 'Server-wide settings: drafting, props, LoRA adapters, slots, models',
			label: 'Control'
		},
		steering: { help: 'Change the steering, which applies to every chat', label: 'Steering' },
		tools: { help: 'Server tools and the MCP proxy', label: 'Tools' }
	};

	const DEFAULT_PERMS: ApiKeyPermission[] = ['chat', 'steering', 'control', 'tools', 'admin'];

	let list = $state<ApiKeysList | null>(null);
	let loadError = $state<string | null>(null);
	let busyId = $state<string | null>(null);

	let newName = $state('');
	let newPerms = $state<ApiKeyPermission[]>(['chat']);
	let creating = $state(false);
	let created = $state<ApiKeyCreated | null>(null);

	let pendingDelete = $state<ApiKeyEntry | null>(null);
	let showDeleteDialog = $state(false);

	let perms = $derived(list?.perms ?? DEFAULT_PERMS);

	onMount(load);

	async function load(): Promise<void> {
		try {
			list = await ApiKeysService.list();
			loadError = null;
		} catch (error) {
			loadError = error instanceof Error ? error.message : 'Failed to load the keys';
		}
	}

	function withPerm(
		current: ApiKeyPermission[],
		perm: ApiKeyPermission,
		on: boolean
	): ApiKeyPermission[] {
		return on ? [...new Set([...current, perm])] : current.filter((existing) => existing !== perm);
	}

	async function createKey(): Promise<void> {
		creating = true;

		try {
			created = await ApiKeysService.create(newName.trim(), newPerms);
			newName = '';
			newPerms = ['chat'];
			await load();
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to create the key');
		} finally {
			creating = false;
		}
	}

	async function update(
		key: ApiKeyEntry,
		patch: Partial<Pick<ApiKeyEntry, 'disabled' | 'name' | 'perms'>>
	): Promise<void> {
		busyId = key.id;

		try {
			await ApiKeysService.update(key.id, patch);
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to change the key');
		} finally {
			// reload either way: on failure it puts the controls back to what the server has
			await load();
			busyId = null;
		}
	}

	function rename(key: ApiKeyEntry, name: string): void {
		if (name.trim() !== key.name) {
			void update(key, { name: name.trim() });
		}
	}

	function askDelete(key: ApiKeyEntry): void {
		pendingDelete = key;
		showDeleteDialog = true;
	}

	async function confirmDelete(): Promise<void> {
		const key = pendingDelete;

		showDeleteDialog = false;
		pendingDelete = null;

		if (!key) return;

		busyId = key.id;

		try {
			await ApiKeysService.remove(key.id);
			toast.success(`Deleted ${key.name || key.id}`);
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Failed to delete the key');
		} finally {
			await load();
			busyId = null;
		}
	}

	function keyLabel(key: ApiKeyEntry): string {
		return key.hint ? `…${key.hint}` : key.id;
	}
</script>

{#snippet permPicker(
	current: ApiKeyPermission[],
	onToggle: (perm: ApiKeyPermission, on: boolean) => void,
	isLocked: (perm: ApiKeyPermission) => boolean,
	idPrefix: string
)}
	<div class="flex flex-wrap gap-x-4 gap-y-2">
		{#each perms as perm (perm)}
			{@const implied = perm !== 'admin' && current.includes('admin')}
			<label
				class="flex cursor-pointer items-center gap-1.5 text-xs select-none"
				for="{idPrefix}-{perm}"
				title={implied ? `${PERM_LABELS[perm].help} (included in admin)` : PERM_LABELS[perm].help}
			>
				<Checkbox
					checked={implied || current.includes(perm)}
					disabled={implied || isLocked(perm)}
					id="{idPrefix}-{perm}"
					onCheckedChange={(on) => onToggle(perm, on === true)}
				/>

				<span>{PERM_LABELS[perm]?.label ?? perm}</span>
			</label>
		{/each}
	</div>
{/snippet}

<SettingsGroup title="API keys">
	{#if loadError}
		<div class="space-y-3">
			<p class="text-sm text-destructive">{loadError}</p>

			<Button onclick={load} size="sm" variant="outline">
				<RefreshCw class="h-3 w-3" />
				Retry
			</Button>
		</div>
	{:else if !list}
		<p class="text-sm text-muted-foreground">Loading…</p>
	{:else}
		<p class="text-sm text-muted-foreground">
			{#if list.key_file}
				The keys live in <code class="text-xs">{list.key_file}</code>. Changes here, or to the file
				itself, apply to the next request without a restart.
			{:else}
				The server was started without <code class="text-xs">--api-key-file</code>, so its keys can
				only change with a restart.
			{/if}
		</p>

		{#if created}
			<div class="space-y-3 rounded-lg border border-border bg-muted/30 p-4">
				<p class="text-sm font-medium">
					New key{created.name ? ` “${created.name}”` : ''}
				</p>

				<div class="flex items-center gap-2">
					<code class="min-w-0 flex-1 truncate rounded bg-muted px-2 py-1 text-xs select-all"
						>{created.key}</code
					>

					<Button
						onclick={() => created && copyToClipboard(created.key, 'Key copied')}
						size="sm"
						variant="outline"
					>
						<Copy class="h-3 w-3" />
						Copy
					</Button>
				</div>

				<p class="text-xs text-muted-foreground">
					Copy it now: the server keeps only its hash and cannot show it again.
				</p>

				<Button onclick={() => (created = null)} size="sm" variant="ghost">Done</Button>
			</div>
		{/if}

		{#if list.key_file}
			<div class="space-y-3">
				<h4 class="text-sm font-medium">New key</h4>

				<Input
					class="w-full"
					maxlength={80}
					oninput={(e) => (newName = e.currentTarget.value)}
					placeholder="Name, for example laptop or ci-runner"
					value={newName}
				/>

				{@render permPicker(
					newPerms,
					(perm, on) => (newPerms = withPerm(newPerms, perm, on)),
					() => false,
					'new-key'
				)}

				<Button disabled={creating} onclick={createKey} size="sm">
					<Plus class="h-3 w-3" />
					Create key
				</Button>
			</div>
		{/if}

		<div class="space-y-3">
			<h4 class="text-sm font-medium">Keys</h4>

			{#if list.keys.length === 0}
				<p class="text-sm text-muted-foreground">No keys yet.</p>
			{/if}

			{#each list.keys as key (key.id)}
				{@const readOnly = key.source === 'cli' || busyId === key.id}

				<div
					class={['space-y-3 rounded-lg border border-border p-3', key.disabled && 'opacity-60']}
				>
					<div class="flex items-center gap-2">
						<KeyRound class="h-4 w-4 shrink-0 text-muted-foreground" />

						{#if key.source === 'file'}
							<Input
								class="h-8 min-w-0 flex-1"
								disabled={readOnly}
								maxlength={80}
								onchange={(e) => rename(key, e.currentTarget.value)}
								placeholder="Unnamed"
								value={key.name}
							/>
						{:else}
							<span class="min-w-0 flex-1 truncate text-sm">{key.name}</span>
						{/if}

						<span class="shrink-0 font-mono text-xs text-muted-foreground" title="Key id {key.id}"
							>{keyLabel(key)}</span
						>

						{#if key.self}
							<Badge variant="secondary">this browser</Badge>
						{/if}
					</div>

					{@render permPicker(
						key.perms,
						(perm, on) => update(key, { perms: withPerm(key.perms, perm, on) }),
						(perm) => readOnly || (key.self === true && perm === 'admin'),
						`key-${key.id}`
					)}

					<div class="flex items-center justify-between gap-2">
						<label
							class="flex cursor-pointer items-center gap-2 text-xs select-none"
							for="enabled-{key.id}"
						>
							<Switch
								checked={!key.disabled}
								disabled={readOnly || key.self === true}
								id="enabled-{key.id}"
								onCheckedChange={(on) => update(key, { disabled: !on })}
							/>

							<span class="text-muted-foreground">{key.disabled ? 'Disabled' : 'Enabled'}</span>
						</label>

						<Button
							disabled={readOnly || key.self === true}
							onclick={() => askDelete(key)}
							size="sm"
							variant="ghost"
						>
							<Trash2 class="h-3 w-3" />
							Delete
						</Button>
					</div>

					{#if key.source === 'cli'}
						<p class="text-xs text-muted-foreground">
							From <code>--api-key</code> on the command line: it changes only with a restart.
						</p>
					{:else if key.self}
						<p class="text-xs text-muted-foreground">
							The key this browser is signed in with cannot lose admin, be disabled or be deleted
							from here.
						</p>
					{/if}
				</div>
			{/each}
		</div>
	{/if}

	{#if !authStore.isAdmin}
		<p class="text-xs text-muted-foreground">Your API key no longer has the admin permission.</p>
	{/if}
</SettingsGroup>

<DialogConfirmation
	bind:open={showDeleteDialog}
	cancelText="Cancel"
	confirmText="Delete"
	description={pendingDelete
		? `Requests that use ${pendingDelete.name || 'this key'} (${keyLabel(pendingDelete)}) will fail from the next one on. This cannot be undone.`
		: ''}
	icon={Trash2}
	onCancel={() => {
		showDeleteDialog = false;
		pendingDelete = null;
	}}
	onConfirm={confirmDelete}
	title="Delete API key"
	variant="destructive"
/>
