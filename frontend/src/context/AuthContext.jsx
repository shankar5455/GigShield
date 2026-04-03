import { createContext, useContext, useState } from 'react';
import { authApi } from '../api';

const AuthContext = createContext(null);

function parseStoredUser(token) {
  if (!token) return null;
  const stored = localStorage.getItem('user');
  if (!stored) return null;
  try {
    return JSON.parse(stored);
  } catch {
    localStorage.removeItem('user');
    return null;
  }
}

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [user, setUser] = useState(() => parseStoredUser(token));
  // loading stays false since user is synchronously initialized from localStorage
  const loading = false;

  const login = async (credentials) => {
    const response = await authApi.login(credentials);
    const { token: newToken, ...userData } = response.data;
    localStorage.setItem('token', newToken);
    localStorage.setItem('user', JSON.stringify(userData));
    setToken(newToken);
    setUser(userData);
    return response.data;
  };

  const register = async (data) => {
    const response = await authApi.register(data);
    const { token: newToken, ...userData } = response.data;
    localStorage.setItem('token', newToken);
    localStorage.setItem('user', JSON.stringify(userData));
    setToken(newToken);
    setUser(userData);
    return response.data;
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
  };

  const isAdmin = () => user?.role === 'ADMIN';
  const isWorker = () => user?.role === 'WORKER';
  const isAuthenticated = () => !!token && !!user;

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout, isAdmin, isWorker, isAuthenticated }}>
      {children}
    </AuthContext.Provider>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};
