import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { CheckCircle, XCircle, Info, X } from 'lucide-react';

const ToastContext = createContext(null);
let _toastCounter = 0;

const ICONS = {
  success: CheckCircle,
  error: XCircle,
  info: Info,
};

const STYLES = {
  success: 'bg-green-50 border-green-200 text-green-800',
  error: 'bg-red-50 border-red-200 text-red-800',
  info: 'bg-blue-50 border-blue-200 text-blue-800',
};

const ICON_STYLES = {
  success: 'text-green-500',
  error: 'text-red-500',
  info: 'text-blue-500',
};

function Toast({ toast, onRemove }) {
  const Icon = ICONS[toast.type] || Info;
  return (
    <div
      className={`flex items-start gap-3 px-4 py-3 rounded-xl border shadow-md text-sm font-medium min-w-[280px] max-w-sm animate-fade-in ${STYLES[toast.type]}`}
    >
      <Icon className={`h-5 w-5 shrink-0 mt-0.5 ${ICON_STYLES[toast.type]}`} />
      <span className="flex-1">{toast.message}</span>
      <button
        onClick={() => onRemove(toast.id)}
        className="shrink-0 opacity-60 hover:opacity-100 transition-opacity"
        aria-label="Dismiss"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

function ToastContainer({ toasts, onRemove }) {
  if (toasts.length === 0) return null;
  return (
    <div className="fixed bottom-4 right-4 z-[200] flex flex-col gap-2">
      {toasts.map((t) => (
        <Toast key={t.id} toast={t} onRemove={onRemove} />
      ))}
    </div>
  );
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const timers = useRef(new Map());

  const removeToast = useCallback((id) => {
    if (timers.current.has(id)) {
      clearTimeout(timers.current.get(id));
      timers.current.delete(id);
    }
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Clear all timers on unmount to avoid setState-after-unmount
  useEffect(() => {
    const timerMap = timers.current;
    return () => timerMap.forEach((t) => clearTimeout(t));
  }, []);

  const addToast = useCallback((message, type = 'info', duration = 4000) => {
    const id = ++_toastCounter;
    setToasts((prev) => [...prev, { id, message, type }]);
    const timer = setTimeout(() => removeToast(id), duration);
    timers.current.set(id, timer);
  }, [removeToast]);

  const toast = {
    success: (msg) => addToast(msg, 'success'),
    error: (msg) => addToast(msg, 'error'),
    info: (msg) => addToast(msg, 'info'),
  };

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}
