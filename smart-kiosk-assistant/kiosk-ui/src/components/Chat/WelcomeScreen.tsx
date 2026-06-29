// Welcome screen for empty state with suggested prompts
interface WelcomeScreenProps {
  onSuggestedPrompt?: (prompt: string) => void;
}

const suggestedPrompts = [
  {
    icon: '🍕',
    title: 'Show Menu',
    prompt: 'What items do you have on the menu?',
  },
  {
    icon: '🛒',
    title: 'Place Order',
    prompt: 'I would like to place an order',
  },
  {
    icon: '⏰',
    title: 'Store Hours',
    prompt: 'What are your opening hours?',
  },
  {
    icon: '❓',
    title: 'Get Help',
    prompt: 'How does this kiosk work?',
  },
];

export function WelcomeScreen({ onSuggestedPrompt }: WelcomeScreenProps) {
  return (
    <div className="kiosk-welcome">
      <div className="kiosk-welcome-header">
        <div className="kiosk-welcome-icon">🤖</div>
        <h2 className="kiosk-welcome-title">Welcome! I'm your kiosk assistant</h2>
        <p className="kiosk-welcome-subtitle">
          Ask me anything about our menu, place an order, or get help
        </p>
      </div>

      <div className="kiosk-welcome-prompts">
        <p className="kiosk-welcome-prompts-label">Try asking:</p>
        <div className="kiosk-welcome-prompts-grid">
          {suggestedPrompts.map((prompt) => (
            <button
              key={prompt.title}
              type="button"
              className="kiosk-welcome-prompt-card"
              onClick={() => onSuggestedPrompt?.(prompt.prompt)}
            >
              <span className="kiosk-welcome-prompt-icon">{prompt.icon}</span>
              <span className="kiosk-welcome-prompt-title">{prompt.title}</span>
            </button>
          ))}
        </div>
      </div>

      <p className="kiosk-welcome-hint">
        <span className="kiosk-welcome-hint-icon">🎤</span>
        Tap the microphone below to start speaking
      </p>
    </div>
  );
}

export default WelcomeScreen;
