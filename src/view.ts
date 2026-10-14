/**
 * Where the board sits in a screenshot: the homography placing its grid, and the squares shown
 * well enough to read. A screenshot looking straight down gets its grid from the period of the
 * tile edges, exact to a negligible fraction of a pixel; one at an angle, from its corners.
 */

import type { Grid } from './grid.js';
import type { Picture } from './picture.js';
import type { Tiles } from './tiles.js';
import type { Homography, Point } from './homography.js';
import type { RGB } from './color.js';

import { EDGE_TOLERANCE, findGrid } from './grid.js';
import { colorDistance } from './color.js';
import { colorAt } from './picture.js';
import { isTileColor, tileShades } from './tiles.js';
import {
	findCorners,
	findPhotographedCorners,
	fitLattice,
	isCornerNear,
	tileClasses,
} from './lattice.js';
import {
	compose,
	fitHomography,
	invert,
	isInFront,
	project,
	raise,
	solveLeastSquares,
	stretch,
} from './homography.js';

// Types -----------------------------------------------------------------------

/** One square of the board. */
export interface Square {
	column: number;
	row: number;
	/**
	 * Its size on screen, in pixels, as the site picks a mipmap level by: across its narrower
	 * extent, which on a board seen at an angle is the one along the line of sight.
	 */
	size: number;
}

/** How a plane on the board's grid sits in a screenshot. */
export interface Plane {
	/** Maps board-grid coordinates to image pixels: square (column, row) spans [column, column+1] x [row, row+1]. */
	toImage: Homography;
	toBoard: Homography;
	/** Camera distortion corrections measured from nearby checkerboard crossings, by square. */
	local?: ReadonlyMap<string, Homography>;
}

/** How the board's grid sits in a screenshot. */
export interface View extends Plane {
	/** The plane the pieces are drawn on, just above the board's. */
	pieces: Plane;
	/** Every square fully on screen and large enough to read. */
	squares: Square[];
	/** Whether the board is seen at an angle. */
	perspective: boolean;
	/** Whether broad application bars surround the board viewport. */
	embedded?: true;
	/** Checkerboard intersections supported by local image evidence, as `column,row`. */
	cornerSupport?: ReadonlySet<string>;
	/** Intersections observed directly, before inferring small gaps in the visible grid. */
	measuredCornerSupport?: ReadonlySet<string>;
	/** A checker of obstacle edge midpoints, when a supported tile-grid alternative was recovered. */
	midpointAlias?: true;
}

// Constants -------------------------------------------------------------------

/** Smallest square size readable, in pixels. */
const MIN_SQUARE_SIZE = 6.5;

/** The fraction of the checkerboard corners a straight-down grid must sit on to be believed. */
const GRID_CORNER_AGREEMENT = 0.6;

/** How few corners make that check meaningless, as on a board crowded by obstacles. */
const MIN_CORNERS_TO_CHECK = 20;

/** The fraction of a lattice's bare square centers that must alternate between the tile colors. */
const CHECKERED_AGREEMENT = 0.9;

/**
 * How far above the board the site draws its pieces at an angle, as a fraction of its camera's
 * height: pieces at 0.005, tiles at -0.01, the camera at 12.
 */
const PIECE_RISE = 0.015 / 12.01;

/** The most squares read at an angle, which keeps a view toward the horizon bounded. */
const MAX_SQUARES = 20000;

// Functions -------------------------------------------------------------------

/**
 * How the board's grid might sit in a screenshot. Seen at an angle, it can take two, one for each
 * way the pieces might stand, for the pieces themselves to tell apart.
 * @throws If there's no checkerboard, or its squares are too small to read.
 */
export function findViews(pic: Picture, tiles: Tiles, embedded = false): View[] {
	const shades = tileShades(pic, tiles);
	const classes = tileClasses(shades);
	let grid: Grid | undefined;
	let gridError: unknown;
	if (!tiles.photographed) {
		try {
			grid = findGrid(pic, shades);
		} catch (error) {
			gridError = error;
		}
	}
	if (grid && !tiles.photographed && areGridCornersShown(pic, classes, grid))
		return [flatView(grid)];
	let corners = findCorners(pic, classes, tiles.photographed ? shades : undefined);
	if (grid && !tiles.photographed && isGridOnCorners(grid, corners)) return [flatView(grid)];
	let homographies = fitLattice(corners, pic.width, pic.height, tiles.photographed, tiles.photographed ? (h, centers) => isCheckerLattice(pic, h, centers, shades) : undefined); // prettier-ignore
	if (tiles.photographed && (!homographies || corners.length < 150)) {
		const localCorners = findPhotographedCorners(pic, Math.max(0.05, Math.min(0.12, 0.65 * colorDistance(tiles[0], tiles[1])))); // prettier-ignore
		const local = fitLattice(
			localCorners,
			pic.width,
			pic.height,
			true,
			(h, centers) => isCheckerLattice(pic, h, centers),
			true,
		);
		if (
			local &&
			(!homographies ||
				(samePhotoGrid(homographies[0]!, local[0]!) &&
					cornerAgreement(local[0]!, localCorners) >
						1.25 * cornerAgreement(homographies[0]!, localCorners)))
		) {
			homographies = local;
			corners = localCorners;
		}
	}
	if (homographies) {
		let photoCrossings: PhotoCrossing[] | undefined;
		const photoContrast = Math.max(
			0.035,
			Math.min(0.1, 0.6 * colorDistance(tiles[0], tiles[1])),
		);
		if (tiles.photographed) {
			const initial = homographies[0]!;
			const refined = refinePhotograph(
				pic,
				perspectiveView(pic, initial, true),
				photoContrast,
			);
			if (refined) {
				const correction = compose(refined.toImage, invert(initial));
				homographies = homographies.map((h) => compose(correction, h));
				photoCrossings = refined.crossings;
			}
		}
		const center = embedded && !tiles.photographed ? viewportCenter(pic, shades) : undefined;
		const views = homographies.map((toImage) => perspectiveView(pic, toImage, tiles.photographed, center, corners, photoCrossings, photoContrast)); // prettier-ignore
		// Closely spaced round obstacles can supply a checkerboard of edge midpoints, rotated
		// halfway between the real axes. Keep the real-cell alternatives for piece shapes to judge.
		if (tiles.photographed && hasAlternatingMarks(pic, views[0]!)) {
			const alternatives: View[] = [];
			for (const h of homographies) {
				for (const phase of [-0.5, 0.5]) {
					const toImage = compose(h, [1, -1, 0.5, 1, 1, phase, 0, 0, 1]);
					const alternative = perspectiveView(pic, toImage, true, undefined, corners);
					if ((alternative.cornerSupport?.size ?? 0) >= 16)
						alternatives.push(alternative);
				}
			}
			if (alternatives.length) {
				for (const view of views) view.midpointAlias = true;
				views.push(...alternatives);
			}
		}
		if (tiles.photographed || isCheckered(pic, tiles, views[0]!, corners)) return views;
	}
	if (grid && !tiles.photographed) return [flatView(grid)];
	throw gridError ?? new Error('No checkerboard with readable squares found in the image.');
}

/** A genuine dark tile stays dark at its rim; a mark mistaken for that tile has a bright rim. */
function hasAlternatingMarks(pic: Picture, view: View): boolean {
	const counts = [0, 0];
	const marks = [0, 0];
	const plain = [0, 0];
	for (const { column, row } of view.squares) {
		if (
			![`${column},${row}`, `${column + 1},${row}`, `${column},${row + 1}`, `${column + 1},${row + 1}`].every((key) => view.cornerSupport?.has(key)) // prettier-ignore
		)
			continue;
		const center = colorNear(pic, view.toImage, column + 0.5, row + 0.5, 2);
		const rim = [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]].map(([x, y]) => colorNear(pic, view.toImage, column + x!, row + y!, 2)); // prettier-ignore
		if (!center || rim.some((color) => !color)) continue;
		const colors = rim as [number, number, number][];
		const background = [0, 1, 2].map((channel) => colors.reduce((sum, color) => sum + color[channel]!, 0) / 4) as [number, number, number]; // prettier-ignore
		if (colors.some((color) => colorDistance(color, background) > 0.08)) continue;
		const parity = (((column + row) % 2) + 2) % 2;
		counts[parity]!++;
		const contrast = colorDistance(center, background);
		if (contrast < 0.08) plain[parity]!++;
		else if (contrast > 0.15 && center.reduce((sum, value) => sum + value, 0) < background.reduce((sum, value) => sum + value, 0) - 0.15) marks[parity]!++; // prettier-ignore
	}
	return [0, 1].some((parity) => counts[parity]! >= 20 && counts[1 - parity]! >= 20 && marks[parity]! > counts[parity]! * 0.6 && plain[1 - parity]! > counts[1 - parity]! * 0.6); // prettier-ignore
}

/**
 * A global palette can hide corners in photographic shadow or glare. Measure local crossings
 * around the preliminary grid and fit again, so visible corners throughout the board calibrate
 * the mapping instead of extrapolating from one bright area.
 */
function refinePhotograph(
	pic: Picture,
	view: View,
	minimumContrast: number,
): { toImage: Homography; crossings: PhotoCrossing[] } | undefined {
	const coordinates = new Set(view.squares.flatMap(({ column, row }) => [
		`${column},${row}`, `${column + 1},${row}`, `${column},${row + 1}`, `${column + 1},${row + 1}`,
	])); // prettier-ignore
	const measured = new Map<string, [Point, Point]>();
	// Choose the stencil at the lattice seed, where its checkerboard was established. A large
	// board can extrapolate to tiny cells outside its viewport, where display marks are noise.
	const size = 1 / stretch(view.toBoard, ...project(view.toImage, 0, 0));
	const small = size < 30;
	const radius = small ? 1 : 2;
	const reaches = small ? [0.16, 0.28] : [0.065, 0.13];
	const colors: RGB[] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]; // prettier-ignore
	for (const key of coordinates) {
		const [column, row] = key.split(',').map(Number) as Point;
		// At small zoom levels a five-pixel average straddles the crossing at the usual reach.
		// Sample farther into each quadrant, using a smaller patch to leave its edges behind.
		let best: Point | undefined;
		let bestQuality = -Infinity;
		for (let dy = -8; dy <= 8; dy++) {
			for (let dx = -8; dx <= 8; dx++) {
				const u = column + dx * 0.025;
				const v = row + dy * 0.025;
				let quality = -0.005 * Math.hypot(dx, dy);
				let checkered = true;
				for (const reach of reaches) {
					const a = colorNear(pic, view.toImage, u - reach, v - reach, radius, colors[0]);
					const b = colorNear(pic, view.toImage, u + reach, v - reach, radius, colors[1]);
					const c = colorNear(pic, view.toImage, u + reach, v + reach, radius, colors[2]);
					const d = colorNear(pic, view.toImage, u - reach, v + reach, radius, colors[3]);
					if (!a || !b || !c || !d) {
						checkered = false;
						break;
					}
					const firstContrast = colorDistance(a, b);
					if (firstContrast < minimumContrast) {
						checkered = false;
						break;
					}
					const contrast = Math.min(firstContrast, colorDistance(c, d));
					const variation = Math.max(colorDistance(a, c), colorDistance(b, d));
					if (variation > 0.1 || contrast < Math.max(minimumContrast, 2 * variation)) {
						checkered = false;
						break;
					}
					quality += contrast - 2 * variation;
				}
				if (checkered && quality > bestQuality) {
					bestQuality = quality;
					best = project(view.toImage, u, v);
				}
			}
		}
		if (best) measured.set(key, [[column, row], refineSmallCrossing(pic, best)]);
	}
	// Isolated marks can resemble a crossing. A measured grid crossing has measured neighbors.
	const pairs = [...measured.values()].filter(([[column, row]]) => [
		`${column - 1},${row}`, `${column + 1},${row}`, `${column},${row - 1}`, `${column},${row + 1}`,
	].filter((key) => measured.has(key)).length >= 2); // prettier-ignore
	if (pairs.length < 12) return undefined;
	const toImage = fitHomography(pairs);
	return toImage.every(Number.isFinite)
		? {
				toImage,
				crossings: pairs.map(([board, measured]) => ({
					ideal: project(toImage, ...board),
					measured,
				})),
			}
		: undefined;
}

/** A crossing retains its grid identity even when curvature moves it toward the next cell. */
interface PhotoCrossing {
	ideal: Point;
	measured: Point;
}

interface PhotoOffset {
	offset: Point;
	weight: number;
	x: number;
	y: number;
}

interface CameraCurve {
	center: Point;
	scale: Point;
	x: number[];
	y: number[];
	precision: number;
}

/** A smooth cubic camera correction, accepted only when most measured crossings agree. */
function photographCurve(view: View, offsets: ReadonlyMap<string, Point>): CameraCurve | undefined {
	if (offsets.size < 80) return undefined;
	const data = [...offsets].map(([key, offset]) => ({
		point: project(view.toImage, ...(key.split(',').map(Number) as Point)),
		offset,
	}));
	let left = Infinity,
		right = -Infinity,
		top = Infinity,
		bottom = -Infinity;
	for (const {
		point: [x, y],
	} of data) {
		left = Math.min(left, x);
		right = Math.max(right, x);
		top = Math.min(top, y);
		bottom = Math.max(bottom, y);
	}
	const center: Point = [(left + right) / 2, (top + bottom) / 2];
	const scale: Point = [(right - left) / 2, (bottom - top) / 2];
	const sizes = view.squares.map((s) => s.size).sort((a, b) => a - b);
	const precision = Math.min(4, 0.12 * sizes[Math.floor(sizes.length / 2)]!);
	const rows = data.map(({ point }) =>
		curveTerms((point[0] - center[0]) / scale[0], (point[1] - center[1]) / scale[1]),
	);
	let weights = data.map(() => 1);
	let x: number[] = [],
		y: number[] = [],
		errors: number[] = [];
	for (let pass = 0; pass < 4; pass++) {
		const weighted = rows.map((row, i) => row.map((v) => v * Math.sqrt(weights[i]!)));
		x = solveLeastSquares(
			weighted,
			data.map((d, i) => d.offset[0] * Math.sqrt(weights[i]!)),
		);
		y = solveLeastSquares(
			weighted,
			data.map((d, i) => d.offset[1] * Math.sqrt(weights[i]!)),
		);
		if (![...x, ...y].every(Number.isFinite)) return undefined;
		errors = rows.map((row, i) =>
			Math.hypot(dot(row, x) - data[i]!.offset[0], dot(row, y) - data[i]!.offset[1]),
		);
		const nextWeights = errors.map((error) => Math.min(1, 2 / Math.max(0.01, error)));
		if (nextWeights.every((weight, i) => weight === weights[i])) break;
		weights = nextWeights;
	}
	errors.sort((a, b) => a - b);
	if (errors[Math.floor(0.9 * errors.length)]! > precision) return undefined;
	return { center, scale, x, y, precision };
}

/** Cubic image coordinates include radial lens curvature and a gently bowed display. */
function curveTerms(x: number, y: number): number[] {
	return [1, x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3];
}

function dot(a: readonly number[], b: readonly number[]): number {
	return a.reduce((sum, value, i) => sum + value * b[i]!, 0);
}

/** The curve supplies the footprint; nearby crossings refine its small remaining translation. */
function curvedPhotographPlane(
	view: View,
	square: Square,
	curve: CameraCurve,
	offsets: ReadonlyMap<string, Point>,
): Homography | undefined {
	const center = project(view.toImage, square.column + 0.5, square.row + 0.5);
	const x = (center[0] - curve.center[0]) / curve.scale[0];
	const y = (center[1] - curve.center[1]) / curve.scale[1];
	const terms = curveTerms(x, y);
	const offset: Point = [dot(terms, curve.x), dot(terms, curve.y)];
	let dx = 0,
		dy = 0,
		total = 0,
		count = 0;
	for (const reach of [1, 4]) {
		dx = dy = total = count = 0;
		for (let r = square.row - reach; r <= square.row + reach + 1; r++) {
			for (let c = square.column - reach; c <= square.column + reach + 1; c++) {
				const actual = offsets.get(`${c},${r}`);
				if (!actual) continue;
				const point = project(view.toImage, c, r);
				const terms = curveTerms(
					(point[0] - curve.center[0]) / curve.scale[0],
					(point[1] - curve.center[1]) / curve.scale[1],
				);
				const rx = actual[0] - dot(terms, curve.x),
					ry = actual[1] - dot(terms, curve.y);
				if (Math.hypot(rx, ry) > curve.precision) continue;
				const weight =
					1 / (0.5 + (c - square.column - 0.5) ** 2 + (r - square.row - 0.5) ** 2);
				dx += rx * weight;
				dy += ry * weight;
				total += weight;
				count++;
			}
		}
		if (count >= 3) break;
	}
	if (count >= 3) {
		offset[0] += dx / total;
		offset[1] += dy / total;
	}
	const displaced = project(view.toBoard, center[0] + offset[0], center[1] + offset[1]);
	if (
		Math.abs(displaced[0] - square.column - 0.5) > 1.25 ||
		Math.abs(displaced[1] - square.row - 0.5) > 1.25
	)
		return undefined;
	const along = [0, 1, 0, 2 * x, y, 0, 3 * x * x, 2 * x * y, y * y, 0];
	const across = [0, 0, 1, 0, x, 2 * y, 0, x * x, 2 * x * y, 3 * y * y];
	const xx = dot(along, curve.x) / curve.scale[0],
		xy = dot(across, curve.x) / curve.scale[1];
	const yx = dot(along, curve.y) / curve.scale[0],
		yy = dot(across, curve.y) / curve.scale[1];
	if (Math.hypot(xx, xy, yx, yy) > 0.4) return undefined;
	return compose(
		[
			1 + xx,
			xy,
			offset[0] - xx * center[0] - xy * center[1],
			yx,
			1 + yy,
			offset[1] - yx * center[0] - yy * center[1],
			0,
			0,
			1,
		],
		view.pieces.toImage,
	);
}

/** Robust local slopes align a curved cell's edges as well as its center. */
function localPhotographPlane(
	view: View,
	square: Square,
	points: PhotoOffset[],
): Homography | undefined {
	if (points.length < 6) return undefined;
	let weights = points.map((p) => p.weight);
	let offset: Point = [0, 0],
		along: Point = [0, 0],
		across: Point = [0, 0];
	const tolerance = Math.max(1.5, 0.04 * square.size);
	for (let pass = 0; pass < 3; pass++) {
		const total = weights.reduce((sum, weight) => sum + weight, 0);
		const mx = points.reduce((sum, p, i) => sum + p.x * weights[i]!, 0) / total;
		const my = points.reduce((sum, p, i) => sum + p.y * weights[i]!, 0) / total;
		const mean = [0, 1].map(
			(c) => points.reduce((sum, p, i) => sum + p.offset[c]! * weights[i]!, 0) / total,
		);
		let xx = 0,
			xy = 0,
			yy = 0;
		const bx: Point = [0, 0],
			by: Point = [0, 0];
		points.forEach((p, i) => {
			const x = p.x - mx,
				y = p.y - my,
				weight = weights[i]!;
			xx += weight * x * x;
			xy += weight * x * y;
			yy += weight * y * y;
			for (const c of [0, 1] as const) {
				bx[c] += weight * x * p.offset[c];
				by[c] += weight * y * p.offset[c];
			}
		});
		const determinant = xx * yy - xy * xy;
		if (determinant <= 0.01 * (xx + yy) ** 2) return undefined;
		along = bx.map((v, c) => (yy * v - xy * by[c]!) / determinant) as Point;
		across = by.map((v, c) => (xx * v - xy * bx[c]!) / determinant) as Point;
		offset = mean.map((v, c) => v - along[c]! * mx - across[c]! * my) as Point;
		const nextWeights = points.map((p) => {
			const error = Math.hypot(
				...p.offset.map((v, c) => v - offset[c]! - along[c]! * p.x - across[c]! * p.y),
			);
			return p.weight * Math.min(1, tolerance / Math.max(0.01, error));
		});
		// Unchanged weights would repeat the identical fit in every remaining pass.
		if (nextWeights.every((weight, i) => weight === weights[i])) break;
		weights = nextWeights;
	}
	const total = weights.reduce((sum, weight) => sum + weight, 0);
	const error =
		points.reduce(
			(sum, p, i) =>
				sum +
				weights[i]! *
					Math.hypot(
						...p.offset.map(
							(v, c) => v - offset[c]! - along[c]! * p.x - across[c]! * p.y,
						),
					),
			0,
		) / total;
	if (error > tolerance || Math.hypot(...along, ...across) > 0.25 * square.size) return undefined;
	const [c, r] = [square.column + 0.5, square.row + 0.5];
	const center = project(view.toImage, c, r);
	const displaced = project(view.toBoard, center[0] + offset[0], center[1] + offset[1]);
	if (Math.abs(displaced[0] - c) > 1.25 || Math.abs(displaced[1] - r) > 1.25) return undefined;
	const a = project(view.toImage, c + 1, r).map((v, i) => v - center[i]!) as Point;
	const b = project(view.toImage, c, r + 1).map((v, i) => v - center[i]!) as Point;
	const determinant = a[0] * b[1] - a[1] * b[0];
	const xx = (along[0] * b[1] - across[0] * a[1]) / determinant;
	const xy = (across[0] * a[0] - along[0] * b[0]) / determinant;
	const yx = (along[1] * b[1] - across[1] * a[1]) / determinant;
	const yy = (across[1] * a[0] - along[1] * b[0]) / determinant;
	return compose(
		[
			1 + xx,
			xy,
			offset[0] - xx * center[0] - xy * center[1],
			yx,
			1 + yy,
			offset[1] - yx * center[0] - yy * center[1],
			0,
			0,
			1,
		],
		view.pieces.toImage,
	);
}

/** Follow neighboring crossings so curvature cannot round an outer crossing into the next cell. */
function connectedPhotographOffsets(view: View, crossings: PhotoCrossing[]): Map<string, Point> {
	const points = crossings.map(({ ideal, measured }) => ({
		point: project(view.toBoard, ...measured),
		nominal: project(view.toBoard, ...ideal).map(Math.round) as Point,
		measured,
		links: [] as { next: number; column: number; row: number }[],
	}));
	const bins = new Map<string, number[]>();
	points.forEach(({ point }, i) => {
		const key = `${Math.floor(point[0])},${Math.floor(point[1])}`;
		const bin = bins.get(key) ?? [];
		bin.push(i);
		bins.set(key, bin);
	});
	points.forEach((p, i) => {
		for (const [column, row] of [
			[1, 0],
			[-1, 0],
			[0, 1],
			[0, -1],
		] as Point[]) {
			for (const step of [1, 2, 3, 4, 5]) {
				const x = p.point[0] + step * column;
				const y = p.point[1] + step * row;
				let closest = -1;
				const reach = step === 1 ? 0.3 : 0.45;
				let error = reach * reach;
				// Only bins intersecting the distance bound can contain an eligible crossing.
				const left = Math.floor(x - reach),
					right = Math.floor(x + reach);
				const top = Math.floor(y - reach),
					bottom = Math.floor(y + reach);
				for (let r = top; r <= bottom; r++) {
					for (let c = left; c <= right; c++) {
						for (const next of bins.get(`${c},${r}`) ?? []) {
							if (next === i) continue;
							const other = points[next]!.point;
							const distance = (other[0] - x) ** 2 + (other[1] - y) ** 2;
							if (distance < error) {
								error = distance;
								closest = next;
							}
						}
					}
				}
				if (closest >= 0) {
					p.links.push({ next: closest, column: column * step, row: row * step });
					break;
				}
			}
		}
	});
	// A gap must agree from both ends; a nearby glyph must not redirect an entire grid branch.
	const links = points.map((p) => p.links);
	points.forEach((p, i) => {
		p.links = p.links.filter((link) =>
			links[link.next]!.some(
				(reverse) =>
					reverse.next === i &&
					reverse.column === -link.column &&
					reverse.row === -link.row,
			),
		);
	});
	const assigned = new Map<number, Point>();
	const conflicts = new Set<number>();
	const offsets = new Map<string, Point>();
	const seeds = points
		.map((_, i) => i)
		.sort((a, b) => {
			const error = (i: number): number => {
				const p = points[i]!;
				return (
					Math.hypot(...p.nominal) +
					0.25 * Math.hypot(p.point[0] - p.nominal[0], p.point[1] - p.nominal[1])
				);
			};
			return error(a) - error(b);
		});
	for (const seed of seeds) {
		if (assigned.has(seed)) continue;
		assigned.set(seed, points[seed]!.nominal);
		const queue = [seed];
		for (let i = 0; i < queue.length; i++) {
			const current = queue[i]!;
			const [c, r] = assigned.get(current)!;
			for (const link of points[current]!.links) {
				const existing = assigned.get(link.next);
				if (existing) {
					if (existing[0] !== c + link.column || existing[1] !== r + link.row) {
						conflicts.add(current);
						conflicts.add(link.next);
					}
					continue;
				}
				assigned.set(link.next, [c + link.column, r + link.row]);
				queue.push(link.next);
			}
		}
		const columns = new Set(queue.map((i) => assigned.get(i)![0]));
		const rows = new Set(queue.map((i) => assigned.get(i)![1]));
		const connected = queue.length >= 12 && columns.size >= 3 && rows.size >= 3;
		for (const i of queue) {
			if (conflicts.has(i)) continue;
			const p = points[i]!;
			const [c, r] = connected ? assigned.get(i)! : p.nominal;
			const limit = connected ? 1.25 : 0.4;
			if (Math.abs(p.point[0] - c) > limit || Math.abs(p.point[1] - r) > limit) continue;
			const key = `${c},${r}`;
			if (!connected && offsets.has(key)) continue;
			const predicted = project(view.toImage, c, r);
			offsets.set(key, [p.measured[0] - predicted[0], p.measured[1] - predicted[1]]);
		}
	}
	return offsets;
}

/** Smooth local crossing residuals correct lens or display curvature beyond one homography. */
function localPhotographPlanes(
	view: View,
	crossings: PhotoCrossing[],
): ReadonlyMap<string, Homography> {
	const local = new Map<string, Homography>();
	const displacements = crossings
		.map(({ ideal, measured }) => Math.hypot(measured[0] - ideal[0], measured[1] - ideal[1]))
		.sort((a, b) => a - b);
	// An already accurate projection needs no warp from occasional glyph-shaped crossings.
	if (!displacements.length || displacements[Math.floor(displacements.length / 2)]! < 0.5)
		return local;
	const offsets = connectedPhotographOffsets(view, crossings);
	const curve = photographCurve(view, offsets);
	if (curve) {
		for (const square of view.squares) {
			const plane = curvedPhotographPlane(view, square, curve, offsets);
			if (plane) local.set(`${square.column},${square.row}`, plane);
		}
		return local;
	}
	for (const square of view.squares) {
		let dx = 0;
		let dy = 0;
		let total = 0;
		let measured = 0;
		let calibrated: Homography | undefined;
		for (const reach of [1, 4]) {
			dx = dy = total = measured = 0;
			const observed: PhotoOffset[] = [];
			for (let r = square.row - reach; r <= square.row + reach + 1; r++) {
				for (let c = square.column - reach; c <= square.column + reach + 1; c++) {
					const offset = offsets.get(`${c},${r}`);
					if (!offset) continue;
					const weight =
						1 / (0.5 + (c - square.column - 0.5) ** 2 + (r - square.row - 0.5) ** 2);
					dx += offset[0] * weight;
					dy += offset[1] * weight;
					total += weight;
					measured++;
					observed.push({
						offset,
						weight,
						x: c - square.column - 0.5,
						y: r - square.row - 0.5,
					});
				}
			}
			calibrated = localPhotographPlane(view, square, observed);
			if (calibrated) break;
			// A glyph corner or a crossing rounded to the next cell can supply an isolated
			// displacement. Keep the strongest group of agreeing crossings before averaging.
			// Smooth neighborhoods need no extra search; only a wide residual spread does.
			const tolerance = Math.max(1.5, 0.1 * square.size);
			if (
				observed.some(
					({ offset }) =>
						Math.hypot(offset[0] - dx / total, offset[1] - dy / total) > 2 * tolerance,
				)
			) {
				let group: typeof observed = [];
				let support = 0;
				for (const { offset } of observed) {
					const agreeing = observed.filter(
						({ offset: other }) =>
							Math.hypot(other[0] - offset[0], other[1] - offset[1]) <= tolerance,
					);
					const weight = agreeing.reduce((sum, point) => sum + point.weight, 0);
					if (agreeing.length >= 3 && weight > support) {
						group = agreeing;
						support = weight;
					}
				}
				measured = group.length;
				dx = dy = total = 0;
				for (const { offset, weight } of group) {
					dx += offset[0] * weight;
					dy += offset[1] * weight;
					total += weight;
				}
			}
			if (measured >= 3) break;
		}
		if (calibrated) {
			local.set(`${square.column},${square.row}`, calibrated);
			continue;
		}
		if (measured < 3) continue;
		dx /= total;
		dy /= total;
		const movement = Math.hypot(dx, dy);
		const center = project(view.toImage, square.column + 0.5, square.row + 0.5);
		const shifted = project(view.toBoard, center[0] + dx, center[1] + dy);
		if (movement < 0.05 || Math.abs(shifted[0] - square.column - 0.5) > 1.25 || Math.abs(shifted[1] - square.row - 0.5) > 1.25) continue; // prettier-ignore
		local.set(
			`${square.column},${square.row}`,
			compose([1, 0, dx, 0, 1, dy, 0, 0, 1], view.pieces.toImage),
		);
	}
	return local;
}

/** The saddle of an averaged crossing locates its edges within the broad quadrant-score peak. */
function refineSmallCrossing(pic: Picture, point: Point): Point {
	const [cx, cy] = point.map(Math.floor) as Point;
	let bx = 0;
	let by = 0;
	let xx = 0;
	let xy = 0;
	let yy = 0;
	const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
	for (let y = -4; y <= 4; y++) {
		for (let x = -4; x <= 4; x++) {
			const color = colorNear(pic, identity, cx + x, cy + y, 2);
			if (!color) return point;
			const value = color[0] + color[1] + color[2];
			bx += value * x;
			by += value * y;
			xx += value * (x * x - 20 / 3);
			xy += value * x * y;
			yy += value * (y * y - 20 / 3);
		}
	}
	[bx, by, xx, xy, yy] = [bx / 540, by / 540, xx / 2772, xy / 3600, yy / 2772];
	const determinant = 4 * xx * yy - xy * xy;
	if (!(determinant < -1e-8)) return point;
	const x = (xy * by - 2 * yy * bx) / determinant;
	const y = (xy * bx - 2 * xx * by) / determinant;
	return Math.hypot(x, y) < 3 ? [cx + 0.5 + x, cy + 0.5 + y] : point;
}

/** The center of a board viewport embedded between broad application bars. */
function viewportCenter(pic: Picture, shades: Float32Array): Point | undefined {
	const rows: number[] = [];
	for (let y = 0; y < pic.height; y++) {
		let dark = 0;
		let light = 0;
		for (let x = 0; x < pic.width; x++) {
			const shade = shades[y * pic.width + x]!;
			if (shade < 0.2) dark++;
			else if (shade > 0.8) light++;
		}
		if (Math.min(dark, light) > pic.width * 0.1) rows.push(y);
	}
	const first = rows[0];
	const last = rows.at(-1);
	if (first === undefined || last === undefined || first < 10 || last > pic.height - 10) return undefined; // prettier-ignore
	return [pic.width / 2, (first + last + 1) / 2];
}

/**
 * A lattice must explain alternating tiles throughout its projected image domain. Restricting
 * the check to its seed's crossings can accept a small periodic moire patch as the whole board.
 * Local color differences keep this test independent of camera exposure and the global palette.
 */
function isCheckerLattice(
	pic: Picture,
	h: Homography,
	centers: Point[],
	shades?: Float32Array,
): boolean {
	let even = 0;
	let odd = 0;
	const nearby = new Set(centers.map(([column, row]) => `${column},${row}`));
	for (const [column, row] of centers) {
		const expected = (((Math.floor(column) + Math.floor(row)) % 2) + 2) % 2 === 0;
		if (shades) {
			const [x, y] = project(h, column, row);
			const shade = shades[Math.floor(y) * pic.width + Math.floor(x)]!;
			if (!Number.isFinite(shade) || (shade > 0.35 && shade < 0.65)) continue;
			if (shade < 0.5 === expected) even++;
			else odd++;
			continue;
		}
		const color = colorNear(pic, h, column, row, 2);
		if (!color) continue;
		const others = [
			[column + 1, row],
			[column - 1, row],
			[column, row + 1],
			[column, row - 1],
		]
			.filter(([c, r]) => nearby.has(`${c},${r}`))
			.map(([c, r]) => colorNear(pic, h, c!, r!, 2))
			.filter((neighbor) => neighbor && colorDistance(color, neighbor) > 0.1);
		if (others.length < 2) continue;
		const brightness = color.reduce((sum, value) => sum + value, 0);
		const darker = others.every(
			(neighbor) => brightness < neighbor!.reduce((sum, value) => sum + value, 0) - 0.1,
		);
		const lighter = others.every(
			(neighbor) => brightness > neighbor!.reduce((sum, value) => sum + value, 0) + 0.1,
		);
		if (!darker && !lighter) continue;
		if (darker === expected) even++;
		else odd++;
	}
	if (even + odd < Math.max(6, centers.length * 0.25) || Math.max(even, odd) < CHECKERED_AGREEMENT * (even + odd)) return false; // prettier-ignore
	return !!shades || !contradictsCheckerDomain(pic, h);
}

/** Many reliable contrary votes reject a local periodic patch; sparse occlusion evidence cannot. */
function contradictsCheckerDomain(pic: Picture, h: Homography): boolean {
	let squares: Square[];
	try {
		squares = perspectiveView(pic, h, true).squares;
	} catch {
		return true;
	}
	const nearby = new Set(squares.map(({ column, row }) => `${column},${row}`));
	let even = 0;
	let odd = 0;
	const stride = Math.max(1, Math.ceil(squares.length / 600));
	for (let index = 0; index < squares.length; index += stride) {
		const { column, row } = squares[index]!;
		const color = colorNear(pic, h, column + 0.5, row + 0.5, 2);
		if (!color) continue;
		const neighbors = [
			[-1, 0],
			[1, 0],
			[0, -1],
			[0, 1],
		]
			.filter(([dx, dy]) => nearby.has(`${column + dx!},${row + dy!}`))
			.map(([dx, dy]) => colorNear(pic, h, column + dx! + 0.5, row + dy! + 0.5, 2))
			.filter((neighbor): neighbor is [number, number, number] => !!neighbor);
		if (neighbors.length < 3) continue;
		const mean = [0, 1, 2].map((channel) => neighbors.reduce((sum, neighbor) => sum + neighbor[channel]!, 0) / neighbors.length) as [number, number, number]; // prettier-ignore
		const contrast = colorDistance(color, mean);
		const variation = Math.max(...neighbors.map((neighbor) => colorDistance(neighbor, mean)));
		if (contrast < 0.1 || variation > 0.45 * contrast) continue;
		const darker = color.reduce((sum, value) => sum + value, 0) < mean.reduce((sum, value) => sum + value, 0); // prettier-ignore
		if (darker === ((((column + row) % 2) + 2) % 2 === 0)) even++;
		else odd++;
	}
	const total = even + odd;
	const expectedErrors = (1 - CHECKERED_AGREEMENT) * total;
	const uncertainty = 3 * Math.sqrt(CHECKERED_AGREEMENT * expectedErrors);
	return Math.min(even, odd) > expectedErrors + uncertainty;
}

/** How many independent local crossings sit on a recovered grid. */
function cornerAgreement(h: Homography, corners: Point[]): number {
	const toBoard = invert(h);
	const supported = new Set<string>();
	for (const corner of corners) {
		const [column, row] = project(toBoard, ...corner);
		const [c, r] = [Math.round(column), Math.round(row)];
		if (Math.abs(column - c) < 0.15 && Math.abs(row - r) < 0.15) supported.add(`${c},${r}`);
	}
	return supported.size;
}

/** A better exposed part can refine a photographed grid without changing its square spacing. */
function samePhotoGrid(reference: Homography, candidate: Homography): boolean {
	const relabeling = compose(invert(candidate), reference);
	const origin = project(relabeling, 0, 0);
	return [
		[1, 0],
		[0, 1],
	].every(([column, row]) => {
		const point = project(relabeling, column!, row!);
		const dx = point[0] - origin[0];
		const dy = point[1] - origin[1];
		const length = Math.hypot(dx, dy);
		return length > 0.8 && length < 1.25 && Math.max(Math.abs(dx), Math.abs(dy)) > 0.9 * length;
	});
}

/**
 * Whether a view's squares alternate between the tile colors at their centers, as a real grid's
 * do. A lattice grown from noise, as squares too small for their corners leave, doesn't.
 */
function isCheckered(pic: Picture, [dark, light]: Tiles, view: View, corners: Point[]): boolean {
	// A small board or separated tile islands can share their palette with application bars.
	// Validate the lattice where its crossings were measured, keeping unrelated UI out of the vote.
	const measured = supportedCorners(view.toBoard, corners).measured;
	const probes: Point[] = [
		[0.12, 0.12],
		[0.88, 0.12],
		[0.12, 0.88],
		[0.88, 0.88],
	];
	let even = 0;
	let odd = 0;
	for (const { column, row } of view.squares) {
		if ([`${column},${row}`, `${column + 1},${row}`, `${column},${row + 1}`, `${column + 1},${row + 1}`].filter((key) => measured.has(key)).length < 3) continue; // prettier-ignore
		// Pieces can occupy every center. Their clear corners still expose the checker tiles.
		let darkVotes = 0;
		let lightVotes = 0;
		for (const [u, v] of probes) {
			const [x, y] = project(view.toImage, column + u, row + v);
			const color = colorAt(pic, Math.floor(y) * pic.width + Math.floor(x));
			if (!isTileColor(color, [dark, light])) continue;
			if (colorDistance(color, dark) < colorDistance(color, light)) darkVotes++;
			else lightVotes++;
		}
		if (darkVotes === lightVotes || Math.max(darkVotes, lightVotes) < 2) continue;
		const isDark = darkVotes > lightVotes;
		if (isDark === ((((column + row) % 2) + 2) % 2 === 0)) even++;
		else odd++;
	}
	return even + odd >= 6 && Math.max(even, odd) >= CHECKERED_AGREEMENT * (even + odd);
}

/** The error for a board whose squares are all too small to read. */
function tooSmall(): Error {
	return new Error(`The squares are under ${MIN_SQUARE_SIZE}px, too small to read. Zoom in.`);
}

/**
 * Whether most of a straight-down grid's inner corners show as checkerboard corners, which settles
 * that the board is seen straight down without scanning the whole image for corners.
 */
function areGridCornersShown(pic: Picture, classes: Int8Array, grid: Grid): boolean {
	let shown = 0;
	let total = 0;
	for (let row = 1; row < grid.y.count; row++) {
		for (let column = 1; column < grid.x.count; column++) {
			const [x, y] = [grid.x.origin + column * grid.size, grid.y.origin + row * grid.size];
			if (isCornerNear(pic, classes, x, y)) shown++;
			total++;
		}
	}
	return total >= MIN_CORNERS_TO_CHECK && shown >= GRID_CORNER_AGREEMENT * total;
}

/** Whether a straight-down grid passes through the checkerboard's corners, as it does unless the board is seen at an angle. */
function isGridOnCorners(grid: Grid, corners: Point[]): boolean {
	if (corners.length < MIN_CORNERS_TO_CHECK) return true;
	const tolerance = Math.max(1, Math.min(2, 0.04 * grid.size));
	const offGrid = (value: number, origin: number): number => {
		const steps = (value - origin) / grid.size;
		return Math.abs(steps - Math.round(steps)) * grid.size;
	};
	const onGrid = corners.filter(
		([x, y]) => offGrid(x, grid.x.origin) < tolerance && offGrid(y, grid.y.origin) < tolerance,
	);
	return onGrid.length >= GRID_CORNER_AGREEMENT * corners.length;
}

/**
 * The view of a board seen straight down.
 * @throws If its squares are too small to read.
 */
function flatView(grid: Grid): View {
	if (grid.size < MIN_SQUARE_SIZE) throw tooSmall();
	const toImage = [grid.size, 0, grid.x.origin, 0, grid.size, grid.y.origin, 0, 0, 1];
	const squares: Square[] = [];
	for (let row = 0; row < grid.y.count; row++) {
		for (let column = 0; column < grid.x.count; column++) squares.push({ column, row, size: grid.size }); // prettier-ignore
	}
	const toBoard = invert(toImage);
	return { toImage, toBoard, pieces: { toImage, toBoard }, squares, perspective: false };
}

/**
 * The view of a board seen at an angle: every readable square connected to the middle of its lattice.
 * @throws If none is large enough to read.
 */
function perspectiveView(
	pic: Picture,
	toImage: Homography,
	photographed = false,
	center?: Point,
	corners?: Point[],
	photoCrossings?: PhotoCrossing[],
	minimumContrast = 0.1,
): View {
	const toBoard = invert(toImage);
	const squares: Square[] = [];
	const seen = new Set<string>();
	const queue: Point[] = [[-1, -1], [0, -1], [-1, 0], [0, 0]]; // prettier-ignore
	for (const [column, row] of queue) seen.add(`${column},${row}`);
	for (let next = 0; next < queue.length && squares.length < MAX_SQUARES; next++) {
		const [column, row] = queue[next]!;
		const square = readableSquare(pic, toImage, toBoard, column, row);
		if (square === undefined) continue;
		squares.push(square);
		for (const [c, r] of [
			[column + 1, row],
			[column - 1, row],
			[column, row + 1],
			[column, row - 1],
		] as Point[]) {
			if (seen.has(`${c},${r}`)) continue;
			seen.add(`${c},${r}`);
			queue.push([c, r]);
		}
	}
	if (squares.length === 0) throw tooSmall();
	const piecesToImage = photographed ? toImage : raise(toImage, pic.width, pic.height, PIECE_RISE, center); // prettier-ignore
	const pieces: Plane = { toImage: piecesToImage, toBoard: invert(piecesToImage) };
	const view: View = {
		toImage,
		toBoard,
		pieces,
		squares,
		perspective: true,
		...(center ? { embedded: true as const } : {}),
	};
	if (photoCrossings) {
		pieces.local = localPhotographPlanes(view, photoCrossings);
		// A corrected neighborhood can expose the next crossing without expanding the search.
		for (let pass = 0; pass < 2; pass++) {
			const recovered = recoverPhotographCrossings(pic, view, minimumContrast);
			if (!recovered.length) break;
			photoCrossings = [...photoCrossings, ...recovered];
			pieces.local = localPhotographPlanes(view, photoCrossings);
		}
		// Calibration can move an outer tile beyond the image. Clamped sampling cannot prove
		// that whole square visible, so apply the original bounds to its corrected footprint.
		view.squares = squares.filter((square) => {
			const plane = pieces.local?.get(`${square.column},${square.row}`);
			return !plane || !!readableSquare(pic, plane, invert(plane), square.column, square.row);
		});
		if (!view.squares.length) throw tooSmall();
	}
	const support = corners && (photographed || center) ? supportedCorners(toBoard, corners, pic, toImage, view.squares, photographed, pieces.local, minimumContrast) : undefined; // prettier-ignore
	view.cornerSupport = support?.supported;
	view.measuredCornerSupport = support?.measured;
	return view;
}

/** Nearby measured curvature can put a hidden crossing within a small second checker search. */
function recoverPhotographCrossings(
	pic: Picture,
	view: View,
	minimumContrast: number,
): PhotoCrossing[] {
	const recovered: PhotoCrossing[] = [];
	const checked = new Set<string>();
	for (const square of view.squares) {
		const corrected = view.pieces.local?.get(`${square.column},${square.row}`);
		if (!corrected) continue;
		for (const [column, row] of [
			[square.column, square.row],
			[square.column + 1, square.row],
			[square.column, square.row + 1],
			[square.column + 1, square.row + 1],
		]) {
			const key = `${column},${row}`;
			if (checked.has(key)) continue;
			checked.add(key);
			if (cornerQuality(pic, corrected, column!, row!, minimumContrast, square.size < 30) > 0)
				continue;
			let best: Point | undefined;
			let quality = 0;
			for (const dy of [-0.125, 0, 0.125]) {
				for (const dx of [-0.125, 0, 0.125]) {
					if (!dx && !dy) continue;
					const u = column! + dx;
					const v = row! + dy;
					const score = cornerQuality(
						pic,
						corrected,
						u,
						v,
						minimumContrast,
						square.size < 30,
					);
					if (score > quality) {
						quality = score;
						best = project(corrected, u, v);
					}
				}
			}
			if (best)
				recovered.push({
					ideal: project(view.toImage, column!, row!),
					measured: refineSmallCrossing(pic, best),
				});
		}
	}
	return recovered;
}

/** Only alternating local quadrants establish a crossing; glyph contrast alone cannot. */
function cornerQuality(
	pic: Picture,
	h: Homography,
	column: number,
	row: number,
	minimumContrast: number,
	small: boolean,
): number {
	let quality = Infinity;
	const radius = small ? 1 : 2;
	for (const reach of small ? [0.16, 0.28] : [0.08, 0.12]) {
		const a = colorNear(pic, h, column - reach, row - reach, radius);
		const b = colorNear(pic, h, column + reach, row - reach, radius);
		const c = colorNear(pic, h, column + reach, row + reach, radius);
		const d = colorNear(pic, h, column - reach, row + reach, radius);
		if (!a || !b || !c || !d) return 0;
		const variation = Math.max(colorDistance(a, c), colorDistance(b, d));
		const contrast = Math.min(colorDistance(a, b), colorDistance(c, d));
		if (variation >= 0.1 || contrast <= Math.max(minimumContrast, 2 * variation)) return 0;
		quality = Math.min(quality, contrast - 2 * variation);
	}
	return quality;
}

/** Detected intersections, supplemented by local contrast where photographic lighting hides them. */
function supportedCorners(
	toBoard: Homography,
	corners: Point[],
	pic?: Picture,
	toImage?: Homography,
	squares?: Square[],
	photographed = false,
	local?: ReadonlyMap<string, Homography>,
	minimumContrast = 0.1,
): { supported: ReadonlySet<string>; measured: ReadonlySet<string> } {
	const supported = new Set<string>();
	let measured: ReadonlySet<string> = supported;
	for (const corner of corners) {
		const [column, row] = project(toBoard, ...corner);
		const [c, r] = [Math.round(column), Math.round(row)];
		if (Math.abs(column - c) < 0.18 && Math.abs(row - r) < 0.18) supported.add(`${c},${r}`); // prettier-ignore
	}
	if (pic && toImage && squares) {
		const checked = new Set<string>();
		for (const { column, row } of squares) {
			for (const [c, r] of [
				[column, row],
				[column + 1, row],
				[column, row + 1],
				[column + 1, row + 1],
			]) {
				const key = `${c},${r}`;
				if (supported.has(key) || checked.has(key)) continue;
				checked.add(key);
				const corrected =
					local?.get(key) ??
					local?.get(`${c! - 1},${r}`) ??
					local?.get(`${c},${r! - 1}`) ??
					local?.get(`${c! - 1},${r! - 1}`) ??
					toImage;
				// Faint checker intersections can establish visibility without steering
				// the more demanding precision fit used for photographed piece sampling.
				if (localCheckerCorner(pic, corrected, c!, r!, photographed, photographed ? Math.min(minimumContrast, 0.035) : minimumContrast)) supported.add(key); // prettier-ignore
			}
		}
		measured = new Set(supported);
		// Fill small interior gaps from a snapshot, without growing one-sided viewport edges.
		for (let pass = 0; pass < 2; pass++) {
			const inferred: string[] = [];
			for (const key of checked) {
				if (supported.has(key)) continue;
				const [c, r] = key.split(',').map(Number) as Point;
				const near = [
					[c + 1, r],
					[c - 1, r],
					[c, r + 1],
					[c, r - 1],
				];
				if (near.filter(([x, y]) => supported.has(`${x},${y}`)).length >= 3) inferred.push(key); // prettier-ignore
			}
			for (const key of inferred) supported.add(key);
		}
		if (photographed) {
			const inferred: string[] = [];
			for (const key of checked) {
				if (supported.has(key)) continue;
				const [c, r] = key.split(',').map(Number) as Point;
				let neighbors = 0;
				for (let dy = -1; dy <= 1; dy++) {
					for (let dx = -1; dx <= 1; dx++) {
						if ((dx || dy) && supported.has(`${c + dx},${r + dy}`)) neighbors++;
					}
				}
				if (neighbors >= 5) inferred.push(key);
			}
			for (const key of inferred) supported.add(key);
		}
	}
	return { supported, measured };
}

/** Local quadrant colors find intersections in bright or shadowed areas outside the global palette. */
function localCheckerCorner(
	pic: Picture,
	toImage: Homography,
	column: number,
	row: number,
	photographed: boolean,
	minimumContrast = 0.1,
): boolean {
	for (const radius of photographed ? [2, 3] : [0, 1]) {
		for (const reach of photographed ? [0.08, 0.12, 0.2] : [0.18, 0.28]) {
			const a = colorNear(pic, toImage, column - reach, row - reach, radius);
			const b = colorNear(pic, toImage, column + reach, row - reach, radius);
			const c = colorNear(pic, toImage, column + reach, row + reach, radius);
			const d = colorNear(pic, toImage, column - reach, row + reach, radius);
			if (!a || !b || !c || !d) continue;
			const variation = Math.max(colorDistance(a, c), colorDistance(b, d));
			const contrast = Math.min(colorDistance(a, b), colorDistance(c, d));
			if (variation < 0.1 && contrast > Math.max(minimumContrast, 2 * variation)) return true;
		}
	}
	return false;
}

/** Average display pixels around a projected point to suppress a photograph's subpixel stripes. */
function colorNear(
	pic: Picture,
	toImage: Homography,
	column: number,
	row: number,
	radius: number,
	out: RGB = [0, 0, 0],
): RGB | undefined {
	const w = toImage[6]! * column + toImage[7]! * row + toImage[8]!;
	const x = Math.floor((toImage[0]! * column + toImage[1]! * row + toImage[2]!) / w);
	const y = Math.floor((toImage[3]! * column + toImage[4]! * row + toImage[5]!) / w);
	if (x < radius || y < radius || x >= pic.width - radius || y >= pic.height - radius) return undefined; // prettier-ignore
	const stride = (pic.width + 1) * 3;
	const side = 2 * radius + 1;
	const a = (y - radius) * stride + (x - radius) * 3;
	const b = a + side * 3;
	const c = a + side * stride;
	const d = c + side * 3;
	const area = side * side;
	out[0] = (pic.sat[d]! - pic.sat[b]! - pic.sat[c]! + pic.sat[a]!) / area;
	out[1] = (pic.sat[d + 1]! - pic.sat[b + 1]! - pic.sat[c + 1]! + pic.sat[a + 1]!) / area;
	out[2] = (pic.sat[d + 2]! - pic.sat[b + 2]! - pic.sat[c + 2]! + pic.sat[a + 2]!) / area;
	return out;
}

/** A square, if it's fully on screen and large enough to read. */
function readableSquare(
	pic: Picture,
	toImage: Homography,
	toBoard: Homography,
	column: number,
	row: number,
): Square | undefined {
	for (const [c, r] of [
		[column, row],
		[column + 1, row],
		[column, row + 1],
		[column + 1, row + 1],
	] as Point[]) {
		if (!isInFront(toImage, c, r)) return undefined;
		const [x, y] = project(toImage, c, r);
		const outside = x < -EDGE_TOLERANCE || y < -EDGE_TOLERANCE || x > pic.width + EDGE_TOLERANCE || y > pic.height + EDGE_TOLERANCE; // prettier-ignore
		if (outside) return undefined;
	}
	const size = 1 / stretch(toBoard, ...project(toImage, column + 0.5, row + 0.5));
	return size >= MIN_SQUARE_SIZE ? { column, row, size } : undefined;
}
