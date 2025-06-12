function evaluateBoard(tempArena) {
    // Heuristique simple : plus de lignes pleines = mieux
    let score = 0;
    for (let y = 0; y < tempArena.length; y++) {
        if (tempArena[y].every(cell => cell !== 0)) score += 100;
    }
    return score;
}

function cloneArena(arena) {
    return arena.map(row => [...row]);
}

function simulateDrop(arena, matrix, x) {
    let y = 0;
    while (!collide(arena, { matrix, pos: { x, y } })) {
        y++;
    }
    y--; // dernière position valide
    const testArena = cloneArena(arena);
    matrix.forEach((row, dy) => {
        row.forEach((val, dx) => {
            if (val !== 0 && testArena[y + dy] && testArena[y + dy][x + dx] !== undefined) {
                testArena[y + dy][x + dx] = val;
            }
        });
    });
    return { arena: testArena, y };
}

function getBestMove() {
    const rotations = [];
    let shape = player.matrix;
    for (let r = 0; r < 4; r++) {
        rotations.push(JSON.parse(JSON.stringify(shape)));
        rotate(shape, 1);
    }

    let bestScore = -Infinity;
    let best = { x: 0, rotation: 0 };

    rotations.forEach((mat, rotIndex) => {
        const width = mat[0].length;
        for (let x = -2; x < arena[0].length - width + 2; x++) {
            const { arena: testArena } = simulateDrop(arena, mat, x);
            const score = evaluateBoard(testArena);
            if (score > bestScore) {
                bestScore = score;
                best = { x, rotation: rotIndex };
            }
        }
    });

    return best;
}

function applyBestMove() {
    const best = getBestMove();

    // Appliquer la rotation
    for (let i = 0; i < best.rotation; i++) {
        playerRotate(1);
    }

    // Déplacer latéralement
    const dx = best.x - player.pos.x;
    if (dx > 0) for (let i = 0; i < dx; i++) playerMove(1);
    if (dx < 0) for (let i = 0; i < -dx; i++) playerMove(-1);

    // Lâcher la pièce
    while (!collide(arena, player)) {
        player.pos.y++;
    }
    player.pos.y--;
    merge(arena, player);
    arenaSweep();
    playerReset();
}

// Remplace playerDrop dans update()
function update(time = 0) {
    const deltaTime = time - lastTime;
    lastTime = time;
    dropCounter += deltaTime;
    if (dropCounter > dropInterval) {
        applyBestMove(); // appel IA ici
        dropCounter = 0;
    }
    draw();
    requestAnimationFrame(update);
}
