/**
 * Sends a command to the server via HTTP GET request.
 * @param {string} command - The command name.
 * @param {string|number} value - The value to send.
 */
function sendCommand(command, value) {
    fetch(`/command/${command}-${value}`, { method: 'GET' })
        .then(() => console.log(`${command} = ${value} sent`))
        .catch(error => console.error('Error:', error));
}

document.addEventListener('DOMContentLoaded', () => {
  const slider = document.getElementById('threshold');
  const value_label = document.getElementById('threshold-value');

  // set initial display
  value_label.textContent = slider.value;

  // live updates as the user drags
  slider.addEventListener('input', () => {
    value_label.textContent = slider.value;
    sendCommand('threshold', slider.value);
  });
});
