(() => {
  const saved = localStorage.getItem('actweave-theme');
  document.documentElement.dataset.theme = saved === 'dark' ? 'dark' : 'light';
  window.setActWeaveTheme = theme => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('actweave-theme', theme);
    const toggle = document.getElementById('theme-toggle');
    toggle.setAttribute('aria-pressed', String(theme === 'dark'));
    toggle.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
  };
  document.addEventListener('DOMContentLoaded', () => {
    setActWeaveTheme(document.documentElement.dataset.theme);
    document.getElementById('theme-toggle').addEventListener('click', () => {
      setActWeaveTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
    });
  });
})();
