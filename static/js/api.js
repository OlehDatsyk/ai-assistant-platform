/**
 * api.js - thin fetch wrapper that attaches the JWT and centralizes error handling.
 * Every page script includes this before its own logic.
 */
const API = {
  base: '',
  token() {
    return localStorage.getItem('access_token') || '';
  },
  setToken(t) {
    localStorage.setItem('access_token', t);
  },
  clearToken() {
    localStorage.removeItem('access_token');
  },
  async request(path, options = {}) {
    const headers = options.headers || {};
    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }
    const token = this.token();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const resp = await fetch(this.base + path, { ...options, headers });
    if (resp.status === 401) {
      this.clearToken();
      window.location.href = '/login';
      throw new Error('Not authenticated');
    }
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const data = await resp.json();
        detail = data.detail || JSON.stringify(data);
      } catch (_) {}
      throw new Error(detail);
    }
    return resp;
  },
  async get(path) {
    const r = await this.request(path, { method: 'GET' });
    return r.json();
  },
  async post(path, body) {
    const r = await this.request(path, {
      method: 'POST',
      body: body instanceof FormData ? body : JSON.stringify(body || {}),
    });
    return r.json();
  },
  async patch(path, body) {
    const r = await this.request(path, { method: 'PATCH', body: JSON.stringify(body || {}) });
    return r.json();
  },
  async put(path, body) {
    const r = await this.request(path, { method: 'PUT', body: JSON.stringify(body || {}) });
    return r.json();
  },
  async del(path) {
    const r = await this.request(path, { method: 'DELETE' });
    return r.json();
  },
};

function requireAuth() {
  if (!API.token() && window.location.pathname !== '/login') {
    window.location.href = '/login';
  }
}
requireAuth();
