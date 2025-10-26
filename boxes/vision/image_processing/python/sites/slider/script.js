/**
 * Sends a command to the server via HTTP GET request.
 * @param {string} command - The command to send.
 */
function sendCommand(command) {
    fetch('/command/' + command, { method: 'GET' })
        .then(response => console.log(`${command} sent`))
        .catch(error => console.error('Error:', error));
}

/**
 * Sets up event listeners for a control button.
 * Supports both mouse and touch inputs.
 * @param {string} id - The button element ID.
 * @param {string} commandOn - The command to send when the button is pressed.
 * @param {string} commandOff - The command to send when the button is released.
 */
function setupButton(id, commandOn, commandOff) {
    let button = document.getElementById(id);

    // Mouse events
    button.addEventListener("mousedown", () => sendCommand(commandOn));
    button.addEventListener("mouseup", () => sendCommand(commandOff));
    button.addEventListener("mouseleave", () => sendCommand(commandOff)); // Stops if user drags finger off

    // Touch events
    button.addEventListener("touchstart", (event) => {
        event.preventDefault(); // Prevents iOS double-tap zoom
        sendCommand(commandOn);
    });

    button.addEventListener("touchend", (event) => {
        event.preventDefault();
        sendCommand(commandOff);
    });
}

// Set up all buttons
setupButton("left", "decrease", "stop");
setupButton("right", "increase", "stop");
