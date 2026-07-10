// Modern typing indicator with animated dots
export function TypingIndicator() {
  return (
    <div className="kiosk-typing-indicator">
      <div className="kiosk-typing-dots">
        <span className="kiosk-typing-dot" style={{ animationDelay: '0ms' }} />
        <span className="kiosk-typing-dot" style={{ animationDelay: '200ms' }} />
        <span className="kiosk-typing-dot" style={{ animationDelay: '400ms' }} />
      </div>
      <span className="kiosk-typing-text">Thinking...</span>
    </div>
  );
}

export default TypingIndicator;
