// browser checks of the web UI against a running server: node ui_e2e.mjs <server url> <steering vector file>
// run from tools/ui (uses its @playwright/test); prints one PASS/FAIL line per check
import { createRequire } from 'module';
import path from 'path';

const require = createRequire(path.join(process.cwd(), 'package.json'));
const { chromium } = require('@playwright/test');
const [U, vec] = process.argv.slice(2);
const res = (ok, name, detail) => console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${detail}`);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
try {
	await page.goto(U + '/');
	await page.waitForTimeout(2500);

	// speculative toggle reflects and changes the server setting
	const spec = await (await fetch(U + '/speculative')).json();
	if (spec.available) {
		const btn = page.getByRole('button', { name: /speculative/i }).first();
		const visible = (await btn.count()) > 0;
		if (visible) {
			await btn.click();
			await page.waitForTimeout(500);
			const after = await (await fetch(U + '/speculative')).json();
			await btn.click();
			await page.waitForTimeout(500);
			const back = await (await fetch(U + '/speculative')).json();
			res(after.enabled !== spec.enabled && back.enabled === spec.enabled, 'speculative toggle', `${spec.enabled} -> ${after.enabled} -> ${back.enabled}`);
		} else {
			res(false, 'speculative toggle', 'button not found');
		}
	}

	// steering dropdown loads a vector and turns it off
	const st = await (await fetch(U + '/steering')).json();
	const item = (st.available || []).find((a) => a.file === vec);
	if (item) {
		const btn = page.getByRole('button', { name: 'Directional steering' }).first();
		await btn.click();
		await page.waitForTimeout(400);
		await page.getByRole('menuitem', { name: new RegExp((item.label || vec).replace(/[.*+?^${}()|[\]\\]/g, '\\$&').slice(0, 12)) }).first().click();
		await page.waitForTimeout(600);
		const on = await (await fetch(U + '/steering')).json();
		await btn.click();
		await page.waitForTimeout(300);
		await page.getByRole('menuitem', { name: /^\s*Off/ }).click();
		await page.waitForTimeout(600);
		const off = await (await fetch(U + '/steering')).json();
		res(on.file === vec && off.file === null, 'steering menu', `loaded ${on.file}, then ${off.file}`);
	}

	// a chat message gets an answer
	const box = page.locator('textarea, [contenteditable="true"]').first();
	await box.click();
	await page.keyboard.type('What is the capital of Australia? One word.');
	await page.keyboard.press('Enter');
	const t0 = Date.now();
	let body = '';
	while (Date.now() - t0 < 180000) {
		await page.waitForTimeout(1000);
		body = (await page.locator('body').innerText()).replace(/\s+/g, ' ');
		if (/Canberra/.test(body.split('Australia? One word.').slice(1).join(' '))) break;
	}
	res(/Canberra/.test(body.split('Australia? One word.').slice(1).join(' ')), 'chat', `answer ${/Canberra/.test(body) ? 'mentions Canberra' : 'missing'} after ${((Date.now() - t0) / 1000).toFixed(0)} s`);
	res(errors.length === 0, 'page errors', errors.length ? errors.slice(0, 2).join(' | ') : 'none');
} catch (e) {
	res(false, 'browser', String(e).slice(0, 200));
} finally {
	await browser.close();
}
