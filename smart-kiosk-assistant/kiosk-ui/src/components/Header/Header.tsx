import BrandSlot from '../../assets/BrandSlot.svg';
import { constants } from '../../constants';

const Header = () => {
  return (
    <header
      className="sticky top-0 left-0 right-0 z-50 bg-intel-blue w-full flex items-center px-4 sm:px-6 border-b border-intel-blue-dark"
      style={{ height: '60px' }}
    >
      {/* Brand */}
      <div className="flex items-center gap-3 sm:gap-4">
        <img src={BrandSlot} alt="Intel" className="h-[44px] w-auto object-contain" />
        <div className="flex flex-col">
          <span className="text-sm sm:text-base font-semibold text-white font-display leading-tight">
            {constants.TITLE}
          </span>
          <span className="text-[10px] text-white/50 font-mono tracking-widest uppercase hidden sm:block">
            AI Kiosk Assistant · v{constants.VERSION}
          </span>
        </div>
      </div>
    </header>
  );
};

export default Header;
