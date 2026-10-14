import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import test from 'node:test';

import { project } from '../src/homography.js';
import { loadPicture } from '../src/picture.js';
import { findTileColors } from '../src/tiles.js';
import { findViews } from '../src/view.js';

test('camera calibration leaves clipped edge squares unread', async () => {
	for (const name of [
		'abundance-photo-gray-wide.png',
		'coaip-no-photo-pink-close.png',
		'pawndard-photo-gray-hand.png',
	]) {
		const pic = await loadPicture(
			await fs.readFile(new URL(`./fixtures/${name}`, import.meta.url)),
		);
		for (const view of findViews(pic, findTileColors(pic))) {
			assert.ok(view.pieces.local?.size, `${name}: exercises camera correction`);
			for (const { column, row } of view.squares) {
				const plane = view.pieces.local?.get(`${column},${row}`) ?? view.pieces.toImage;
				for (const [c, r] of [
					[column, row],
					[column + 1, row],
					[column, row + 1],
					[column + 1, row + 1],
				]) {
					const [x, y] = project(plane, c!, r!);
					assert.ok(
						x >= -1 && y >= -1 && x <= pic.width + 1 && y <= pic.height + 1,
						`${name}: square ${column},${row} has a clipped corner at ${x},${y}`,
					);
				}
			}
		}
	}
});
