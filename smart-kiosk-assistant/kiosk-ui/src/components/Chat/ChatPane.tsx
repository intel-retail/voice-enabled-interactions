import { useEffect, useRef, useState } from 'react';
import type { ChatMessage, VoicePhase } from '../../types';

interface ChatPaneProps {
  messages: ChatMessage[];
  partialUser: string;
  partialAssistant: string;
  phase: VoicePhase;
}

function Bubble({
  role,
  text,
  streaming,
  isLatest,
}: {
  role: 'user' | 'assistant';
  text: string;
  streaming?: boolean;
  isLatest?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const isUser = role === 'user';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  };

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} kiosk-message-fade-in`}
      style={
        isLatest
          ? {
              animation: 'messageSlideIn 0.3s ease-out',
            }
          : undefined
      }
    >
      <div
        className={`group relative max-w-[80%] rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap break-words shadow-sm transition-all duration-150 ${
          isUser
            ? 'bg-kiosk-user text-white rounded-br-sm hover:shadow-md'
            : 'bg-kiosk-asst text-intel-dark rounded-bl-sm hover:shadow-md'
        }`}
      >
        {text}
        {streaming ? <span className="kiosk-cursor ml-0.5 inline-block">▋</span> : null}

        {!isUser && !streaming && text && (
          <button
            type="button"
            onClick={() => void handleCopy()}
            className="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-white rounded-full p-1.5 shadow-md border border-kiosk-border hover:bg-kiosk-pane"
            title="Copy message"
            aria-label="Copy message to clipboard"
          >
            {copied ? (
              <svg className="w-3.5 h-3.5 text-green-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-3.5 h-3.5 text-intel-blue" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
            )}
          </button>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="bg-kiosk-asst rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
        <div className="flex items-center space-x-1">
          <div className="kiosk-typing-dot" style={{ animationDelay: '0ms' }} />
          <div className="kiosk-typing-dot" style={{ animationDelay: '200ms' }} />
          <div className="kiosk-typing-dot" style={{ animationDelay: '400ms' }} />
        </div>
      </div>
    </div>
  );
}

function WelcomeScreen() {
  const suggestions = [
    { icon: '🍔', text: '"What\'s on the menu?"' },
    { icon: '🍕', text: '"Show me your burgers"' },
    { icon: '🛒', text: '"I\'d like to order a burger"' },
    { icon: '⏰', text: '"What are your opening hours?"' },
  ];

  return (
    <div className="h-full flex flex-col items-center justify-center text-center px-6">
      <div className="text-6xl mb-4 animate-bounce-slow">🤖</div>
      <h2 className="text-2xl font-semibold text-intel-dark mb-2">Welcome! I'm your kiosk assistant</h2>
      <p className="text-kiosk-textmd mb-8 max-w-md">Ask me anything about our menu, place an order, or get help</p>

      <div className="mb-8">
        <p className="text-xs text-kiosk-textlo uppercase tracking-wide font-medium mb-3">Try asking:</p>
        <div className="grid grid-cols-2 gap-3">
          {suggestions.map((s, i) => (
            <div
              key={i}
              className="flex items-center space-x-2 bg-white rounded-lg border border-kiosk-border px-4 py-3 text-sm text-kiosk-textmd hover:border-intel-blue hover:bg-kiosk-pane transition-all duration-150 cursor-pointer"
            >
              <span className="text-lg">{s.icon}</span>
              <span>{s.text}</span>
            </div>
          ))}
        </div>
      </div>

      <p className="text-xs text-kiosk-textlo flex items-center space-x-2">
        <span className="text-lg">🎤</span>
        <span>Tap the microphone below to start speaking</span>
      </p>
    </div>
  );
}

/**
 * Scrollable conversation view. Renders finalized turns plus any in-progress
 * partial user transcript / assistant response with a blinking cursor.
 */
export function ChatPane({ messages, partialUser, partialAssistant, phase }: ChatPaneProps) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, partialUser, partialAssistant]);

  const showPartialUser = (phase === 'listening' || phase === 'processing') && !!partialUser;
  const showPartialAssistant = phase === 'processing' && !!partialAssistant;
  const showTyping = phase === 'processing' && !partialAssistant && messages.length > 0;
  const empty = messages.length === 0 && !showPartialUser && !showPartialAssistant;

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {empty ? <WelcomeScreen /> : null}

      {messages.map((m, i) => (
        <Bubble key={i} role={m.role} text={m.text} isLatest={i === messages.length - 1} />
      ))}

      {showPartialUser ? <Bubble role="user" text={partialUser} streaming /> : null}
      {showPartialAssistant ? <Bubble role="assistant" text={partialAssistant} streaming /> : null}
      {showTyping ? <TypingIndicator /> : null}

      <div ref={endRef} />
    </div>
  );
}

export default ChatPane;
