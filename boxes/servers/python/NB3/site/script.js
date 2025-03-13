const container = document.querySelector(".container");

let lastX = 0, lastY = 0, isDragging = false;

// Function to update shadow and movement
function updateShadowAndMovement(x, y) {
    const normalizedX = (x / window.innerWidth - 0.5) * 2;
    const normalizedY = (y / window.innerHeight - 0.5) * 2;

    const shadowOffsetX = -normalizedX * 100;
    const shadowOffsetY = -normalizedY * 100;

    const moveOffsetX = normalizedX * 30;
    const moveOffsetY = normalizedY * 30;

    container.style.boxShadow = `${shadowOffsetX}px ${shadowOffsetY}px 25px rgba(0, 0, 0, 0.5)`;
    container.style.transform = `translate(${moveOffsetX}px, ${moveOffsetY}px)`;
}

// Handle mouse movement
document.addEventListener("mousemove", (event) => {
    updateShadowAndMovement(event.clientX, event.clientY);
});

// Handle touch start
document.addEventListener("touchstart", (event) => {
    isDragging = true;
    lastX = event.touches[0].clientX;
    lastY = event.touches[0].clientY;
});

// Handle touch move
document.addEventListener("touchmove", (event) => {
    if (!isDragging) return;

    const touchX = event.touches[0].clientX;
    const touchY = event.touches[0].clientY;

    updateShadowAndMovement(touchX, touchY);

    lastX = touchX;
    lastY = touchY;
});

// Handle touch end
document.addEventListener("touchend", () => {
    isDragging = false;
});
