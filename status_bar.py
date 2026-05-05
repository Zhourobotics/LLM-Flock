"""
Status Bar Module

Provides a persistent status bar at the bottom of the terminal that shows
simulation progress while allowing normal logging output above.
"""

from blessed import Terminal
import atexit

class StatusBar:
    """Persistent status bar that stays locked at the bottom of the terminal."""
    
    def __init__(self):
        self.term = Terminal()
        self.is_active = False
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def start(self):
        """Initialize the status bar."""
        self.is_active = True
        # Hide cursor for cleaner display
        print(self.term.hide_cursor, end="", flush=True)
    
    def update(self, test_name, current_round, total_rounds, mode="", additional_info=""):
        """Update the status bar with current simulation state."""
        if not self.is_active:
            return
            
        # Create status message
        status_msg = f"Test: {test_name} | Round: {current_round}/{total_rounds}"
        if mode:
            status_msg += f" | Mode: {mode}"
        if additional_info:
            status_msg += f" | {additional_info}"
        
        # Save cursor position, move to bottom, print status, restore cursor
        print(
            self.term.save +
            self.term.move_yx(self.term.height - 1, 0) +
            self.term.on_blue + self.term.white +
            status_msg.ljust(self.term.width - 1) +
            self.term.normal +
            self.term.restore,
            end="",
            flush=True
        )
    
    def cleanup(self):
        """Clean up the status bar on exit."""
        if self.is_active:
            # Clear the status bar and show cursor
            print(
                self.term.move_yx(self.term.height - 1, 0) +
                self.term.clear_eol +
                self.term.show_cursor,
                end="",
                flush=True
            )
            self.is_active = False

# Global status bar instance
status_bar = StatusBar()