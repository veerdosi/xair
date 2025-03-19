import React from 'react';

const Badge = ({ variant = 'default', className = '', ...props }) => {
  const variants = {
    default: 'bg-gray-100 text-gray-900',
    primary: 'bg-blue-100 text-blue-900',
    secondary: 'bg-gray-100 text-gray-900',
    destructive: 'bg-red-100 text-red-900',
    outline: 'border border-gray-200 text-gray-900',
  };

  const variantClasses = variants[variant] || variants.default;

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${variantClasses} ${className}`}
      {...props}
    />
  );
};

export { Badge };

// === Slider.jsx ===
import React from 'react';

const Slider = ({
  value = [0],
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  onValueChange,
  className = '',
  ...props
}) => {
  const handleChange = (e) => {
    const newValue = [parseInt(e.target.value, 10)];
    onValueChange?.(newValue);
  };

  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value[0]}
      disabled={disabled}
      onChange={handleChange}
      className={`h-2 w-full cursor-pointer appearance-none rounded-full bg-gray-200 ${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      } ${className}`}
      style={{
        /* Custom slider styling */
        '--track-bg': 'rgb(229 231 235)',
        '--thumb-bg': 'white',
        '--thumb-border': 'rgb(59 130 246)',
        
        /* Webkit styles */
        WebkitAppearance: 'none',
      }}
      {...props}
    />
  );
};

export { Slider };