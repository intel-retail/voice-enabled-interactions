import BrandSlot from '../../assets/BrandSlot.svg';
import { constants } from '../../constants';
import type { VoicePhase } from '../../types';

interface HeaderProps {
  phase: VoicePhase;
}

function PhaseChip({ phase }: { phase: VoicePhase }) {
  const map: Record<VoicePhase, { label: string; cls: string }> = {
    idle:       { label: 'READY',        cls: 'bg-white/10 text-white/60' },
    listening:  { label: '● LISTENING',  cls: 'bg-red-500/20 text-red-300 animate-pulse' },
    processing: { label: '⟳ PROCESSING', cls: 'bg-amber-500/20 text-amber-300' },
    speaking:   { label: '▶ SPEAKING',   cls: 'bg-green-500/20 text-green-300' },
  };
  const { label, cls } = map[phase];
  return (
    <div className={`rounded-full px-3 py-1 text-[11px] font-bold tracking-widest ${cls}`}>
      {label}
    </div>
  );
}

const Header = ({ phase }: HeaderProps) => {
  return (
    <header
      className="sticky top-0 left-0 right-0 z-50 bg-intel-blue w-full flex items-center justify-between px-6 border-b border-intel-blue-dark"
      style={{ height: '60px' }}
    >
      {/* Left: Brand */}
      <div className="flex items-center gap-4">
        <img src={BrandSlot} alt="Intel" className="h-[48px] w-auto object-contain" />
        <div className="flex flex-col">
          <span className="text-base font-semibold text-white font-display leading-tight">
            {constants.TITLE}
          </span>
          <span className="text-[10px] text-white/50 font-mono tracking-widest uppercase">
            AI Benchmarking Demo · v{constants.VERSION}
          </span>
        </div>
      </div>

      {/* Right: Phase chip */}
      <PhaseChip phase={phase} />
    </header>
  );
};

export default Header;
