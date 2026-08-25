'use strict';

/* AIRMAN landing page motion.
 *
 * Two rules govern everything below.
 *
 * 1. Motion never carries meaning alone. Every animation is a gsap.from() or
 *    writes a value the page already has in markup, so if the GSAP CDN is
 *    blocked the page still renders complete and readable. The only elements
 *    that start empty (dial arcs, comparison bars) get an explicit
 *    no-GSAP fallback in applyStaticFallback().
 * 2. Scrolling is a descent. The HUD altitude tape unwinds FL410 -> ground
 *    across the document, which is why the tape is driven by total scroll
 *    progress rather than by any one section.
 */

const $ = (id) => document.getElementById(id);
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const HAS_GSAP = typeof window.gsap !== 'undefined';

/* ------------------------------------------------------------------ */
/* Live service status — the HUD reports the real backend, not a mock. */
/* ------------------------------------------------------------------ */
async function loadHealth() {
  const dot = $('statusDot');
  const text = $('statusText');
  const foot = $('footHealth');
  try {
    const res = await fetch('/health');
    const body = await res.json();
    const vectors = body.index && body.index.vector_loaded ? 'VECTOR+LEXICAL' : 'LEXICAL ONLY';
    const path = (body.generation_path || 'unknown').toUpperCase();
    dot.dataset.state = body.status === 'ok' ? 'ok' : 'degraded';
    text.textContent = `${body.status.toUpperCase()} · ${vectors} · ${path}`;
    foot.textContent = `${vectors} · GEN ${path}`;
  } catch (err) {
    dot.dataset.state = 'down';
    text.textContent = 'SERVICE UNREACHABLE';
    foot.textContent = 'SERVICE UNREACHABLE';
  }
}

/* ------------------------------------------------------------------ */
/* Hero starfield                                                      */
/* ------------------------------------------------------------------ */
function startSky() {
  const canvas = $('sky');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let stars = [];
  let beacons = [];
  let raf = null;

  function size() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
  }

  function seed() {
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    const count = Math.round((w * h) / 9000);
    stars = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.25 + 0.25,
      a: Math.random() * 0.5 + 0.12,
      // Twinkle rate; slow enough to read as atmosphere rather than noise.
      tw: Math.random() * 0.011 + 0.002,
      ph: Math.random() * Math.PI * 2,
      // Depth drives both parallax and drift, so far stars move less.
      z: Math.random() * 0.7 + 0.3,
    }));
    beacons = Array.from({ length: 3 }, (_, i) => ({
      x: Math.random() * w,
      y: h * (0.16 + i * 0.2),
      speed: (Math.random() * 0.18 + 0.09) * (Math.random() < 0.5 ? -1 : 1),
      ph: Math.random() * Math.PI * 2,
    }));
  }

  let mx = 0, my = 0, tx = 0, ty = 0;
  window.addEventListener('mousemove', (e) => {
    tx = (e.clientX / window.innerWidth - 0.5) * 22;
    ty = (e.clientY / window.innerHeight - 0.5) * 14;
  });

  let t = 0;
  function frame() {
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    t += 1;
    mx += (tx - mx) * 0.045;
    my += (ty - my) * 0.045;

    ctx.clearRect(0, 0, w, h);

    for (const s of stars) {
      const alpha = s.a * (0.55 + 0.45 * Math.sin(s.ph + t * s.tw));
      // Slow leftward drift: the aircraft is moving, so the sky is too.
      s.x -= 0.028 * s.z;
      if (s.x < -4) s.x = w + 4;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = alpha > 0.42 ? '#FFD795' : '#BBB6B0';
      ctx.beginPath();
      ctx.arc(s.x + mx * s.z, s.y + my * s.z, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Distant traffic: anti-collision beacons, one flash per ~1.4 s.
    for (const b of beacons) {
      b.x += b.speed;
      if (b.x > w + 30) b.x = -30;
      if (b.x < -30) b.x = w + 30;
      const flash = Math.pow(Math.max(0, Math.sin(b.ph + t * 0.045)), 14);
      if (flash > 0.01) {
        ctx.globalAlpha = flash * 0.85;
        ctx.fillStyle = '#E29019';
        ctx.beginPath();
        ctx.arc(b.x + mx * 0.4, b.y + my * 0.4, 1.9, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(frame);
  }

  size();
  window.addEventListener('resize', size);

  if (REDUCED) {
    // One static frame: the sky is present, it simply does not move.
    frame();
    cancelAnimationFrame(raf);
    return;
  }
  frame();

  // Stop burning frames when the hero is off screen or the tab is hidden.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { cancelAnimationFrame(raf); }
    else { raf = requestAnimationFrame(frame); }
  });
}

/* ------------------------------------------------------------------ */
/* HUD tapes                                                           */
/* ------------------------------------------------------------------ */
function buildHeadingTape() {
  const strip = $('hdgStrip');
  if (!strip) return;
  const frag = document.createDocumentFragment();
  // Two full turns so the strip can translate continuously without a seam.
  for (let pass = 0; pass < 2; pass++) {
    for (let deg = 0; deg < 360; deg += 10) {
      const b = document.createElement('b');
      const label = String(deg / 10).padStart(2, '0');
      b.textContent = deg % 90 === 0 ? ['N', 'E', 'S', 'W'][deg / 90] : label;
      frag.appendChild(b);
      const i = document.createElement('i');
      frag.appendChild(i);
    }
  }
  strip.appendChild(frag);
}

function wireHudToScroll() {
  const altValue = $('altValue');
  const altFill = $('altFill');
  const altPhase = $('altPhase');
  const strip = $('hdgStrip');

  function update() {
    const max = document.body.scrollHeight - window.innerHeight;
    const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;

    const alt = Math.round((41000 * (1 - p)) / 25) * 25;
    altValue.textContent = alt.toLocaleString('en-US');
    altFill.style.height = (p * 100) + '%';
    altPhase.textContent = p < 0.15 ? 'CRUISE' : p < 0.85 ? 'DESCENT' : 'SHORT FINAL';

    if (strip) strip.style.transform = `translateX(${-p * 1400}px)`;
  }

  update();
  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
}

/* ------------------------------------------------------------------ */
/* Fallback for the two components that start empty in markup          */
/* ------------------------------------------------------------------ */
function applyStaticFallback() {
  document.querySelectorAll('.gauge-cell').forEach((cell) => {
    const val = parseFloat(cell.dataset.val) || 0;
    const arc = cell.querySelector('.dial-arc');
    const len = 2 * Math.PI * 50;
    arc.style.strokeDasharray = String(len);
    arc.style.strokeDashoffset = String(len * (1 - val / 100));
    cell.querySelector('.num').textContent = val.toFixed(val % 1 === 0 ? 0 : 1);
  });
  document.querySelectorAll('.cmp-row').forEach((row) => {
    row.querySelector('.bar.a i').style.width = row.dataset.a + '%';
    row.querySelector('.bar.b i').style.width = row.dataset.b + '%';
  });
  document.querySelectorAll('[data-count]').forEach((el) => {
    el.textContent = Number(el.dataset.count).toLocaleString('en-US');
  });
}

/* ------------------------------------------------------------------ */
/* GSAP timelines                                                      */
/* ------------------------------------------------------------------ */
function animate() {
  gsap.registerPlugin(ScrollTrigger, MotionPathPlugin);
  gsap.defaults({ ease: 'power3.out' });

  /* --- Hero entrance ------------------------------------------------ */
  const intro = gsap.timeline({ delay: 0.15 });
  intro
    .from('.hero-eyebrow', { opacity: 0, x: -18, duration: 0.7 })
    .from('#wordmark .ch', {
      yPercent: 115,
      opacity: 0,
      duration: 1.05,
      stagger: 0.055,
      ease: 'expo.out',
    }, '-=0.35')
    .from('#heroLede', { opacity: 0, y: 22, duration: 0.8 }, '-=0.55')
    .from('#heroActions .btn', { opacity: 0, y: 16, duration: 0.6, stagger: 0.08 }, '-=0.45')
    .from('#scrollCue', { opacity: 0, duration: 0.6 }, '-=0.3');

  gsap.to('.cue-dot', {
    y: 46, duration: 1.5, repeat: -1, ease: 'power1.inOut', opacity: 0,
  });

  // Hero recedes as the page scrolls past it, so the sections below feel
  // like they are arriving rather than sliding over a static backdrop.
  gsap.to('#heroInner', {
    y: -90, opacity: 0, ease: 'none',
    scrollTrigger: { trigger: '#hero', start: 'top top', end: 'bottom top', scrub: 0.4 },
  });
  gsap.to('.horizon-glow', {
    y: -160, ease: 'none',
    scrollTrigger: { trigger: '#hero', start: 'top top', end: 'bottom top', scrub: 0.6 },
  });

  /* --- Section tags ------------------------------------------------- */
  gsap.utils.toArray('.sec-tag').forEach((tag) => {
    gsap.from(tag, {
      opacity: 0, x: -14, duration: 0.6,
      scrollTrigger: { trigger: tag, start: 'top 88%' },
    });
  });

  /* --- The rule: words land one at a time --------------------------- */
  splitWords($('ruleCopy'));
  gsap.from('#ruleCopy .w', {
    opacity: 0, yPercent: 60, rotateX: -55, duration: 0.75, stagger: 0.035,
    scrollTrigger: { trigger: '#ruleCopy', start: 'top 78%' },
  });
  gsap.from('#refusalCard', {
    opacity: 0, y: 30, duration: 0.8,
    scrollTrigger: { trigger: '#refusalCard', start: 'top 85%' },
  });
  gsap.from('.refusal-bar', {
    scaleY: 0, transformOrigin: 'top', duration: 0.9, ease: 'power2.inOut',
    scrollTrigger: { trigger: '#refusalCard', start: 'top 85%' },
  });

  /* --- Flight plan: route draws, aircraft flies it ------------------ */
  const path = $('routePath');
  const plane = $('aircraft');
  if (path) {
    const len = path.getTotalLength();
    gsap.set(path, { strokeDasharray: len, strokeDashoffset: len });
    gsap.set(plane, { opacity: 0 });

    const waypoints = gsap.utils.toArray('.wp');
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: '.pipeline',
        start: 'top 62%',
        end: 'bottom 75%',
        scrub: 0.7,
        onUpdate: (self) => {
          // Waypoints light as the aircraft passes them. Their x positions in
          // the viewBox are roughly evenly spaced, so progress is a fair proxy.
          waypoints.forEach((wp, i) => {
            wp.classList.toggle('lit', self.progress >= i / waypoints.length);
          });
        },
      },
    });
    tl.to(path, { strokeDashoffset: 0, ease: 'none', duration: 1 }, 0)
      .to(plane, { opacity: 1, duration: 0.05 }, 0)
      .to(plane, {
        motionPath: { path: path, align: path, alignOrigin: [0.5, 0.5], autoRotate: 90 },
        ease: 'none', duration: 1,
      }, 0);
  }

  gsap.from('.stg', {
    opacity: 0, y: 34, duration: 0.7, stagger: 0.09,
    scrollTrigger: { trigger: '.stages-grid', start: 'top 85%' },
  });

  /* --- Instrument dials --------------------------------------------- */
  const CIRC = 2 * Math.PI * 50;
  gsap.utils.toArray('.gauge-cell').forEach((cell, i) => {
    const target = parseFloat(cell.dataset.val) || 0;
    const arc = cell.querySelector('.dial-arc');
    const num = cell.querySelector('.num');
    gsap.set(arc, { strokeDasharray: CIRC, strokeDashoffset: CIRC });

    const counter = { v: 0 };
    gsap.timeline({ scrollTrigger: { trigger: '.gauge-row', start: 'top 80%' }, delay: i * 0.11 })
      .to(arc, {
        strokeDashoffset: CIRC * (1 - target / 100),
        duration: 1.5, ease: 'power2.inOut',
      }, 0)
      .to(counter, {
        v: target, duration: 1.5, ease: 'power2.inOut',
        onUpdate: () => { num.textContent = counter.v.toFixed(target % 1 === 0 ? 0 : 1); },
      }, 0);
  });
  gsap.from('.gauge-cell', {
    opacity: 0, y: 26, duration: 0.7, stagger: 0.08,
    scrollTrigger: { trigger: '.gauge-row', start: 'top 86%' },
  });

  /* --- Comparison bars ---------------------------------------------- */
  gsap.utils.toArray('.cmp-row').forEach((row) => {
    const tl = gsap.timeline({ scrollTrigger: { trigger: row, start: 'top 86%' } });
    tl.from(row, { opacity: 0, x: -16, duration: 0.5 }, 0)
      .to(row.querySelector('.bar.a i'), {
        width: row.dataset.a + '%', duration: 0.9, ease: 'power2.out',
      }, 0.1)
      .to(row.querySelector('.bar.b i'), {
        width: row.dataset.b + '%', duration: 1.1, ease: 'power2.out',
      }, 0.2);
  });

  /* --- Corpus marquee: scroll velocity nudges it -------------------- */
  const track = $('mqTrack');
  if (track) {
    // Duplicate the run so the loop has no visible seam.
    track.innerHTML += track.innerHTML;
    const half = track.scrollWidth / 2;
    const drift = gsap.to(track, {
      x: -half, duration: 26, ease: 'none', repeat: -1,
      modifiers: { x: (x) => (parseFloat(x) % half) + 'px' },
    });
    ScrollTrigger.create({
      trigger: '.corpus',
      start: 'top bottom',
      end: 'bottom top',
      onUpdate: (self) => {
        // Scrolling down speeds the strip up; scrolling up reverses it.
        const v = gsap.utils.clamp(-4, 4, self.getVelocity() / 320);
        gsap.to(drift, { timeScale: v === 0 ? 1 : v, duration: 0.4, overwrite: true });
      },
    });
  }

  /* --- Counters ------------------------------------------------------ */
  gsap.utils.toArray('[data-count]').forEach((el) => {
    const target = Number(el.dataset.count);
    const obj = { v: 0 };
    gsap.to(obj, {
      v: target, duration: 1.8, ease: 'power2.out',
      scrollTrigger: { trigger: el, start: 'top 90%' },
      onUpdate: () => { el.textContent = Math.round(obj.v).toLocaleString('en-US'); },
    });
  });
  gsap.from('.corpus-stats div', {
    opacity: 0, y: 24, duration: 0.6, stagger: 0.07,
    scrollTrigger: { trigger: '.corpus-stats', start: 'top 88%' },
  });

  /* --- CTA ----------------------------------------------------------- */
  gsap.to('#sweep', { rotate: 360, duration: 5.5, repeat: -1, ease: 'none' });
  gsap.from('.cta-h', {
    opacity: 0, y: 40, duration: 0.9,
    scrollTrigger: { trigger: '.cta', start: 'top 72%' },
  });
  gsap.from('.cta-p, .cta .btn', {
    opacity: 0, y: 20, duration: 0.7, stagger: 0.12,
    scrollTrigger: { trigger: '.cta', start: 'top 68%' },
  });
  gsap.from('.cta-radar span', {
    scale: 0.7, opacity: 0, duration: 1.1, stagger: 0.12, ease: 'power2.out',
    scrollTrigger: { trigger: '.cta', start: 'top 78%' },
  });
}

/* Wrap each word of an element in a span so it can be staggered. */
function splitWords(el) {
  if (!el) return;
  const walk = (node) => {
    Array.from(node.childNodes).forEach((child) => {
      if (child.nodeType === Node.TEXT_NODE) {
        const frag = document.createDocumentFragment();
        child.textContent.split(/(\s+)/).forEach((piece) => {
          if (!piece.trim()) { frag.appendChild(document.createTextNode(piece)); return; }
          const span = document.createElement('span');
          span.className = 'w';
          span.textContent = piece;
          frag.appendChild(span);
        });
        node.replaceChild(frag, child);
      } else if (child.nodeType === Node.ELEMENT_NODE) {
        walk(child);
      }
    });
  };
  walk(el);
}

/* ------------------------------------------------------------------ */
buildHeadingTape();
wireHudToScroll();
startSky();
loadHealth();

if (HAS_GSAP && !REDUCED) {
  animate();
} else {
  // No GSAP, or motion is suppressed: paint every final state directly.
  applyStaticFallback();
}
