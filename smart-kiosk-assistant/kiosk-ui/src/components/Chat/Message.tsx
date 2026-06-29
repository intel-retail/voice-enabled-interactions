// Individual message component with animations and modern styling
import { useState } from 'react';
import type { ChatMessage } from '../../types';

interface MessageProps {
  message: ChatMessage;
  isLatest?: boolean;
}

export function Message({ message, isLatest }: MessageProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  };

  return (
    <div
      className={`kiosk-message kiosk-message-${message.role} ${
        isLatest ? 'kiosk-message-latest' : ''
      }`}
      style={{
        animation: 'messageSlideIn 0.3s ease-out',
      }}
    >
      <div className="kiosk-message-content-wrapper">
        <div className={`kiosk-message-content ${isUser ? 'kiosk-message-user' : 'kiosk-message-assistant'}`}>
          <div className="kiosk-message-text">{message.text}</div>
          
          {!isUser && message.text && (
            <button
              type="button"
              className="kiosk-copy-btn"
              onClick={() => void handleCopy()}
              title="Copy message"
              aria-label="Copy message to clipboard"
            >
              {copied ? (
                <svg className="kiosk-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="kiosk-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
              )}
              <span className="kiosk-copy-label">{copied ? 'Copied!' : 'Copy'}</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default Message;
