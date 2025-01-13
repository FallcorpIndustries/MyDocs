// brailleAnimation.js

// Grab all Braille cells
// IMPORTANT: Must match the classes in your HTML: .braille-item
// so the querySelectorAll() will find them.
function generateBrailleAnimations() {
    const cells = document.querySelectorAll(".braille-item");

    // Braille patterns (A-Z)
    const brailleAlphabet = {
      A: [[0, 0]],
      B: [[0, 0], [0, 1]],
      C: [[0, 0], [1, 0]],
      D: [[0, 0], [1, 0], [1, 1]],
      E: [[0, 0], [1, 1]],
      F: [[0, 0], [0, 1], [1, 0]],
      G: [[0, 0], [0, 1], [1, 0], [1, 1]],
      H: [[0, 0], [0, 1], [1, 1]],
      I: [[0, 1], [1, 0]],
      J: [[0, 1], [1, 0], [1, 1]],
      K: [[0, 0], [2, 0]],
      L: [[0, 0], [0, 1], [2, 0]],
      M: [[0, 0], [1, 0], [2, 0]],
      N: [[0, 0], [1, 0], [1, 1], [2, 0]],
      O: [[0, 0], [1, 1], [2, 0]],
      P: [[0, 0], [0, 1], [1, 0], [2, 0]],
      Q: [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]],
      R: [[0, 0], [0, 1], [1, 1], [2, 0]],
      S: [[0, 1], [1, 0], [2, 0]],
      T: [[0, 1], [1, 0], [1, 1], [2, 0]],
      U: [[0, 0], [2, 0], [2, 1]],
      V: [[0, 0], [0, 1], [2, 0], [2, 1]],
      W: [[0, 1], [1, 0], [1, 1], [2, 1]],
      X: [[0, 0], [1, 0], [2, 0], [2, 1]],
      Y: [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]],
      Z: [[0, 0], [1, 1], [2, 0], [2, 1]],
    };

    function generateBraillePattern(cell, letter) {
      // Fade out existing dots
      const existingDots = cell.querySelectorAll(".dot");
      existingDots.forEach((dot) => {
        dot.classList.add("disappear");
        setTimeout(() => dot.remove(), 500);
      });

      // Braille pattern for the letter
      const pattern = brailleAlphabet[letter];

      // Add new dots
      pattern.forEach(([row, col]) => {
        const dot = document.createElement("div");
        dot.classList.add("dot");
        // Approx. position each dot
        dot.style.top = `${row * 30 + 15}px`;
        dot.style.left = `${col * 30 + 15}px`;
        // Random short delay for appear
        setTimeout(() => dot.classList.add("appear"), Math.random() * 300);
        cell.appendChild(dot);
      });
    }

    function updateBraillePatterns() {
      cells.forEach((cell) => {
        // Random letter
        const randomLetter = Object.keys(brailleAlphabet)[Math.floor(Math.random() * 26)];
        generateBraillePattern(cell, randomLetter);
      });
    }

    // Refresh Braille patterns every 1 second
    setInterval(updateBraillePatterns, 1000);

    // Initial run
    updateBraillePatterns();
}

document.addEventListener("DOMContentLoaded", generateBrailleAnimations);
