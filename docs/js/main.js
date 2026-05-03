/* =============================================================================
   YADEM — JavaScript Interactions
   ============================================================================= */

document.addEventListener('DOMContentLoaded', () => {

  // --- Navbar scroll effect ---
  const navbar = document.querySelector('.navbar');
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 40);
  });

  // --- Mobile hamburger ---
  const hamburger = document.querySelector('.nav-hamburger');
  const navLinks = document.querySelector('.nav-links');
  if (hamburger) {
    hamburger.addEventListener('click', () => {
      navLinks.classList.toggle('active');
    });
    // Close menu on link click
    navLinks.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => navLinks.classList.remove('active'));
    });
  }

  // --- Scroll fade-in observer ---
  const fadeEls = document.querySelectorAll('.fade-in, .pipeline-step');
  const fadeObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 80);
        fadeObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });
  fadeEls.forEach(el => fadeObserver.observe(el));

  // --- Counter animation ---
  const counterEls = document.querySelectorAll('[data-count]');
  const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        counterObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });
  counterEls.forEach(el => counterObserver.observe(el));

  function animateCounter(el) {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || '';
    const prefix = el.dataset.prefix || '';
    const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
    const duration = 2000;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = eased * target;
      el.textContent = prefix + current.toFixed(decimals) + suffix;
      if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }

  // --- Score meter animation ---
  const meters = document.querySelectorAll('.score-meter-fill');
  const meterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const target = entry.target.dataset.width || '68';
        setTimeout(() => { entry.target.style.width = target + '%'; }, 300);
        meterObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });
  meters.forEach(el => meterObserver.observe(el));

  // --- Smooth scroll for anchor links ---
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        const offset = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--nav-height')) || 72;
        window.scrollTo({ top: target.offsetTop - offset, behavior: 'smooth' });
      }
    });
  });

  // --- Typing effect for hero ---
  const typingEl = document.querySelector('.typing-text');
  if (typingEl) {
    const phrases = ['Credit Scoring', 'Risk Assessment', 'Fraud Detection', 'Financial Inclusion'];
    let phraseIndex = 0;
    let charIndex = 0;
    let deleting = false;

    function type() {
      const current = phrases[phraseIndex];
      if (deleting) {
        typingEl.textContent = current.substring(0, charIndex--);
        if (charIndex < 0) { deleting = false; phraseIndex = (phraseIndex + 1) % phrases.length; setTimeout(type, 500); return; }
      } else {
        typingEl.textContent = current.substring(0, charIndex++);
        if (charIndex > current.length) { deleting = true; setTimeout(type, 2000); return; }
      }
      setTimeout(type, deleting ? 40 : 80);
    }
    setTimeout(type, 1200);
  }

  // --- API Demo: live request simulation ---
  const demoBtn = document.getElementById('demo-run-btn');
  if (demoBtn) {
    demoBtn.addEventListener('click', () => {
      const responseEl = document.getElementById('demo-response');
      responseEl.innerHTML = '<div style="text-align:center;padding:40px;color:var(--accent-cyan);">⏳ Scoring applicant...</div>';
      demoBtn.disabled = true;
      demoBtn.textContent = 'Processing...';

      setTimeout(() => {
        const score = Math.floor(Math.random() * 400 + 500);
        const band = score >= 800 ? 'A' : score >= 650 ? 'B' : score >= 500 ? 'C' : 'D';
        const bandMeaning = { A: 'Excellent', B: 'Good', C: 'Borderline', D: 'High Risk' }[band];
        const decision = score >= 650 ? 'AUTO_APPROVED' : score >= 500 ? 'MANUAL_REVIEW' : 'DECLINED';
        const decisionClass = decision === 'AUTO_APPROVED' ? 'badge-approved' : decision === 'MANUAL_REVIEW' ? 'badge-band' : '';
        const pd = (1 - score / 1000).toFixed(4);
        const time = (Math.random() * 150 + 80).toFixed(0);

        responseEl.innerHTML = `
          <div class="response-item"><span class="response-label">YADEM Score</span><span class="response-value" style="color:var(--accent-cyan);font-size:1.4rem;">${score}</span></div>
          <div class="response-item"><span class="response-label">Risk Band</span><span class="${decisionClass || 'badge-band'}">${band} — ${bandMeaning}</span></div>
          <div class="response-item"><span class="response-label">Decision</span><span class="${decision === 'AUTO_APPROVED' ? 'badge-approved' : 'badge-band'}">${decision}</span></div>
          <div class="response-item"><span class="response-label">Default Probability</span><span class="response-value">${pd}</span></div>
          <div class="response-item"><span class="response-label">Processing Time</span><span class="response-value">${time}ms</span></div>
          <div class="score-meter"><div class="score-meter-fill" data-width="${score / 10}" style="width:0"></div></div>
        `;

        // Animate the meter
        setTimeout(() => {
          const fill = responseEl.querySelector('.score-meter-fill');
          if (fill) fill.style.width = (score / 10) + '%';
        }, 100);

        demoBtn.disabled = false;
        demoBtn.textContent = '▶ Run Again';
      }, 1800);
    });
  }

});
