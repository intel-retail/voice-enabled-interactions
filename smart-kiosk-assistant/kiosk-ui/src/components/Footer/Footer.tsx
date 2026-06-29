import { constants } from '../../constants';

const Footer = () => {
  return (
    <footer className="sticky bottom-0 left-0 right-0 w-full bg-intel-blue text-white text-center px-8 h-12 text-sm z-10 shadow-[0_-2px_8px_rgba(0,0,0,0.04)] border-t border-intel-blue-dark flex items-center justify-center font-text">
      <span>{constants.COPYRIGHT}</span>
    </footer>
  );
};

export default Footer;
