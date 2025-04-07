/**
 * Client-side JavaScript file.
 * Includes logic for auto-resizing the query textarea.
 */

function autoResize(textarea) {
    // Reset height to auto to shrink if text is deleted
    textarea.style.height = 'auto';
    // Set height to the scroll height to fit content
    textarea.style.height = textarea.scrollHeight + 'px';
  }
  
  document.addEventListener('DOMContentLoaded', function() {
      // Add event listener to the textarea for resizing on input
      const queryTextarea = document.getElementById('query');
      if (queryTextarea) {
          // Initial resize on page load in case there's pre-filled text
          autoResize(queryTextarea);
  
          // Add listener for input events
          queryTextarea.addEventListener('input', function() {
              autoResize(this);
          });
      }
  });
  
  