import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import test from 'node:test';
import sharp from 'sharp';

import { loadPicture } from '../src/picture.js';
import { findTileColors } from '../src/tiles.js';

const FIXTURES = new URL('./fixtures/', import.meta.url);

// Capture type is independently labeled in the fixture manifest, never inferred from the reader.
const fixtures: { image: string; photographed?: true }[] = JSON.parse(
	await fs.readFile(new URL('fixtures.json', FIXTURES), 'utf8'),
);
const PHOTOS = new Set(
	fixtures.filter((fixture) => fixture.photographed).map((fixture) => fixture.image),
);

test('keeps every fixture screenshot out of camera processing', async () => {
	const screenshots = (await fs.readdir(FIXTURES)).filter(
		(name) => name.endsWith('.png') && !PHOTOS.has(name),
	);
	for (const name of screenshots) {
		const tiles = findTileColors(await loadPicture(await fs.readFile(new URL(name, FIXTURES))));
		assert.equal(tiles.photographed, undefined, name);
	}
});

test('recognizes camera captures despite their different palettes and framing', async () => {
	for (const name of PHOTOS) {
		const tiles = findTileColors(await loadPicture(await fs.readFile(new URL(name, FIXTURES))));
		assert.equal(tiles.photographed, true, name);
	}
});

test('preserves screenshot processing after resizing, JPEG compression and contrast reduction', async () => {
	for (const name of [
		'blue-wide.png',
		'wood-wide.png',
		'space-tilted.png',
		'persp-palace-gray.png',
		'persp-coaip-ro-lime-fullwindow.png',
		'persp-4x4x4x4-chess-gray-starfield.png',
	]) {
		const source = await fs.readFile(new URL(name, FIXTURES));
		const variants = [
			['JPEG', await sharp(source).jpeg({ quality: 60 }).toBuffer()],
			['resized', await sharp(source).resize({ width: 731 }).png().toBuffer()],
			[
				'resized JPEG',
				await sharp(source).resize({ width: 731 }).jpeg({ quality: 60 }).toBuffer(),
			],
			['low contrast', await sharp(source).linear(0.3, 120).png().toBuffer()],
		] as const;
		for (const [variant, buffer] of variants) {
			const tiles = findTileColors(await loadPicture(buffer));
			assert.equal(tiles.photographed, undefined, `${name}: ${variant}`);
		}
	}
	// Dense round obstacles hide the plain tile interiors, and resizing blends their outlines.
	const dense = await sharp(await fs.readFile(new URL('obstocean-crop-tight.png', FIXTURES)))
		.resize({ width: 731 })
		.png()
		.toBuffer();
	assert.equal(
		findTileColors(await loadPicture(dense)).photographed,
		undefined,
		'dense obstacles',
	);
});
