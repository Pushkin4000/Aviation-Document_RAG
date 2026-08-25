'use strict';

/* AIRMAN landing page motion.
 *
 * MOTION POLICY
 * -------------
 * The first version of this file had a single hard branch: if the browser
 * reported `prefers-reduced-motion: reduce`, it skipped GSAP entirely and
 * painted final states. That is wrong twice over.
 *
 * First, it over-reads the signal. Windows sets that flag whenever
 * "Animation effects" is off in Settings > Accessibility > Visual effects,
 * which many machines have off for performance rather than for vestibular
 * sensitivity. A large share of visitors therefore got a completely static
 * page that looked broken.
 *
 * Second, reduced motion does not mean *no* motion. The guidance is to remove
 * what triggers vestibular symptoms — large parallax, spins, scale changes,
 * and scroll-jacked scrubbing — not to remove all feedback. Opacity fades and
 * small movements are explicitly fine.
 *
 * So motion runs in two tiers:
 *
 *   full     everything: parallax, scrubbed flight path, starfield, marquee,
 *            radar sweep, reticle, magnetic buttons.
 *   reduced  fades and <10px moves only. Content still animates in, counters
 *            still count, the route still draws — nothing translates far,
 *            spins, scales, or follows the scrollbar.
 *
 * Tier is chosen by explicit override first (localStorage, set by the on-page
 * toggle), then by the OS preference. The toggle exists so a visitor whose OS
 * flag does not reflect what they actually want is one click from the other
 * tier, in either direction.
 *
 * FAILURE POLICY
 * --------------
 * Motion never carries meaning alone. Every entrance is a `gsap.from()`, so
 * the markup's own state is the final state and a blocked CDN leaves a
 * complete page. The two components that genuinely start empty — dial arcs
 * and comparison bars — are painted by applyStaticFallback() whenever GSAP or
 * ScrollTrigger is unavailable.
 */

const $ = (id) => document.getElementById(id);

const MOTION_KEY = 'airman-motion';
const OS_REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const HAS_GSAP = typeof window.gsap !== 'undefined';
const HAS_ST = HAS_GSAP && typeof window.ScrollTrigger !== 'undefined';
const HAS_MP = HAS_GSAP && typeof window.MotionPathPlugin !== 'undefined';

function storedMotion() {
  try { return localStorage.getItem(MOTION_KEY); } catch (err) { return null; }
}
function queryMotion() {
  const q = new URLSearchParams(window.location.search).get('motion');
  return q === 'full' || q === 'reduced' ? q : null;
}

// ?motion=full / ?motion=reduced wins over everything, so either tier can be
// linked to directly and verified in a headless browser (which reports
// `reduce` unconditionally and would otherwise be untestable).
const OVERRIDE = queryMotion() || storedMotion();
const MOTION = OVERRIDE === 'full' || OVERRIDE === 'reduced'
  ? OVERRIDE
  : (OS_REDUCED ? 'reduced' : 'full');
const FULL = MOTION === 'full';

/* ------------------------------------------------------------------ */
/* Motion toggle                                                       */
/* ------------------------------------------------------------------ */
function wireMotionToggle() {
  const btn = $('motionToggle');
  const label = $('motionLabel');
  if (!btn) return;
  const paint = () => {
    btn.setAttribute('aria-pressed', String(FULL));
    label.textContent = FULL ? 'MOTION: FULL' : 'MOTION: REDUCED';
    btn.title = FULL
      ? 'Full motion. Click for reduced motion.'
      : 'Reduced motion' + (OVERRIDE ? '' : ' (following your system setting)') + '. Click for full motion.';
  };
  paint();
  btn.addEventListener('click', () => {
    try { localStorage.setItem(MOTION_KEY, FULL ? 'reduced' : 'full'); } catch (err) { /* private mode */ }
    // Reload rather than rebuild: tearing down dozens of live ScrollTriggers
    // and re-running them correctly is far more failure-prone than a reload.
    window.location.reload();
  });
}

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
  let running = false;

  function seed() {
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    const count = Math.round((w * h) / 9000);
    stars = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.25 + 0.25,
      a: Math.random() * 0.5 + 0.12,
      tw: Math.random() * 0.011 + 0.002,
      ph: Math.random() * Math.PI * 2,
      z: Math.random() * 0.7 + 0.3,
    }));
    beacons = Array.from({ length: 3 }, (_, i) => ({
      x: Math.random() * w,
      y: h * (0.16 + i * 0.2),
      speed: (Math.random() * 0.18 + 0.09) * (Math.random() < 0.5 ? -1 : 1),
      ph: Math.random() * Math.PI * 2,
    }));
  }

  function size() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
    if (!FULL) draw(0);
  }

  let mx = 0, my = 0, tx = 0, ty = 0;
  if (FULL) {
    window.addEventListener('mousemove', (e) => {
      tx = (e.clientX / window.innerWidth - 0.5) * 22;
      ty = (e.clientY / window.innerHeight - 0.5) * 14;
    });
  }

  function draw(t) {
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    ctx.clearRect(0, 0, w, h);

    for (const s of stars) {
      const alpha = FULL ? s.a * (0.55 + 0.45 * Math.sin(s.ph + t * s.tw)) : s.a;
      if (FULL) {
        s.x -= 0.028 * s.z;
        if (s.x < -4) s.x = w + 4;
      }
      ctx.globalAlpha = alpha;
      ctx.fillStyle = alpha > 0.42 ? '#FFD795' : '#BBB6B0';
      ctx.beginPath();
      ctx.arc(s.x + mx * s.z, s.y + my * s.z, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    if (FULL) {
      // Distant traffic: anti-collision beacons, roughly one flash per 1.4 s.
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
    }
    ctx.globalAlpha = 1;
  }

  let t = 0;
  function frame() {
    t += 1;
    mx += (tx - mx) * 0.045;
    my += (ty - my) * 0.045;
    draw(t);
    raf = requestAnimationFrame(frame);
  }

  function start() { if (!running) { running = true; raf = requestAnimationFrame(frame); } }
  function stop() { running = false; cancelAnimationFrame(raf); }

  size();
  window.addEventListener('resize', size);

  if (!FULL) { draw(0); return; }

  start();
  document.addEventListener('visibilitychange', () => (document.hidden ? stop() : start()));
  // The canvas only exists inside the hero; below the fold it is pure waste.
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((entries) => {
      entries[0].isIntersecting ? start() : stop();
    }, { threshold: 0 }).observe($('hero'));
  }
}

/* ------------------------------------------------------------------ */
/* HUD tapes                                                           */
/* ------------------------------------------------------------------ */
function buildHeadingTape() {
  const strip = $('hdgStrip');
  if (!strip) return;
  const frag = document.createDocumentFragment();
  // Two full turns so the strip can translate without exposing a seam.
  for (let pass = 0; pass < 2; pass++) {
    for (let deg = 0; deg < 360; deg += 10) {
      const b = document.createElement('b');
      b.textContent = deg % 90 === 0 ? ['N', 'E', 'S', 'W'][deg / 90] : String(deg / 10).padStart(2, '0');
      frag.appendChild(b);
      frag.appendChild(document.createElement('i'));
    }
  }
  strip.appendChild(frag);
}

function wireHudToScroll() {
  const altValue = $('altValue');
  const altFill = $('altFill');
  const altPhase = $('altPhase');
  const strip = $('hdgStrip');
  const adi = $('adiFace');
  let bank = 0, bankTarget = 0;

  if (FULL && adi) {
    window.addEventListener('mousemove', (e) => {
      bankTarget = (e.clientX / window.innerWidth - 0.5) * 16;
    });
  }

  function update() {
    const max = document.body.scrollHeight - window.innerHeight;
    const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;

    const alt = Math.round((41000 * (1 - p)) / 25) * 25;
    altValue.textContent = alt.toLocaleString('en-US');
    altFill.style.height = (p * 100) + '%';
    altPhase.textContent = p < 0.15 ? 'CRUISE' : p < 0.85 ? 'DESCENT' : 'SHORT FINAL';

    // Reduced tier keeps the readouts — they are information — but holds the
    // tape and the attitude ball still, since both are scroll-driven motion.
    if (strip && FULL) strip.style.transform = `translateX(${-p * 1400}px)`;
    if (adi && FULL) {
      const pitch = -8 + p * 30;
      adi.style.transform = `rotate(${bank}deg) translateY(${pitch}px)`;
    }
  }

  if (FULL && adi) {
    (function ease() {
      bank += (bankTarget - bank) * 0.06;
      update();
      requestAnimationFrame(ease);
    })();
  } else {
    update();
  }

  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
}

/* ------------------------------------------------------------------ */
/* Cursor reticle + magnetic buttons (full tier only)                  */
/* ------------------------------------------------------------------ */
function wirePointerFlourishes() {
  if (!FULL || !HAS_GSAP || window.matchMedia('(hover: none)').matches) return;

  const ret = $('reticle');
  if (ret) {
    const qx = gsap.quickTo(ret, 'x', { duration: 0.35, ease: 'power3' });
    const qy = gsap.quickTo(ret, 'y', { duration: 0.35, ease: 'power3' });
    window.addEventListener('mousemove', (e) => {
      gsap.to(ret, { opacity: 1, duration: 0.3, overwrite: 'auto' });
      qx(e.clientX);
      qy(e.clientY);
    });
    document.addEventListener('mouseleave', () => gsap.to(ret, { opacity: 0, duration: 0.3 }));
  }

  document.querySelectorAll('.magnetic').forEach((el) => {
    const strength = 0.28;
    el.addEventListener('mousemove', (e) => {
      const r = el.getBoundingClientRect();
      gsap.to(el, {
        x: (e.clientX - (r.left + r.width / 2)) * strength,
        y: (e.clientY - (r.top + r.height / 2)) * strength,
        duration: 0.4, ease: 'power3.out',
      });
    });
    el.addEventListener('mouseleave', () => {
      gsap.to(el, { x: 0, y: 0, duration: 0.6, ease: 'elastic.out(1, 0.4)' });
    });
  });
}

/* ------------------------------------------------------------------ */
/* Text scramble — instrument-panel decode                             */
/* ------------------------------------------------------------------ */
const GLYPHS = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789/·#*';

function scramble(el, duration) {
  const final = el.dataset.text || el.textContent;
  const total = Math.round((duration || 900) / 16);
  let frame = 0;
  const seeds = Array.from(final, (ch, i) => ({
    ch,
    start: Math.round((i / final.length) * total * 0.6),
    end: Math.round((i / final.length) * total * 0.6) + 8 + Math.random() * 10,
  }));

  return new Promise((resolve) => {
    (function tick() {
      let out = '';
      let done = 0;
      for (const s of seeds) {
        if (s.ch === ' ') { out += ' '; done++; continue; }
        if (frame >= s.end) { out += s.ch; done++; }
        else if (frame >= s.start) { out += GLYPHS[Math.floor(Math.random() * GLYPHS.length)]; }
        else { out += ' '; }
      }
      el.textContent = out;
      if (done === seeds.length) { el.textContent = final; resolve(); return; }
      frame++;
      requestAnimationFrame(tick);
    })();
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
/* Fallback for the components that start empty in markup              */
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
  document.querySelectorAll('.wp').forEach((wp) => wp.classList.add('lit'));
  const plane = $('aircraft');
  if (plane) plane.style.opacity = '0';
}

/* ------------------------------------------------------------------ */
/* Boot sequence                                                       */
/* ------------------------------------------------------------------ */
function runBoot(onDismissed) {
  const boot = $('boot');
  const bar = $('bootBar');
  const lines = Array.from(document.querySelectorAll('.boot-log li'));
  if (!boot) { onDismissed(); return; }

  let dismissed = false;
  const dismiss = () => {
    if (dismissed) return;
    dismissed = true;
    boot.classList.add('done');
    document.body.style.overflow = '';
    onDismissed();
  };

  document.body.style.overflow = 'hidden';
  const step = FULL ? 190 : 60;
  const holdFor = lines.length * step + 220;

  lines.forEach((li, i) => {
    setTimeout(() => {
      li.classList.add('on');
      if (bar) {
        bar.style.transition = `width ${step}ms linear`;
        bar.style.width = (((i + 1) / lines.length) * 100) + '%';
      }
    }, i * step);
  });

  setTimeout(() => {
    if (HAS_GSAP && FULL) {
      gsap.to(boot, { yPercent: -100, duration: 0.75, ease: 'expo.inOut', onComplete: dismiss });
    } else {
      dismiss();
    }
  }, holdFor);

  /* WATCHDOG — load-bearing, not belt-and-braces.
     The overlay covers the page and locks body scroll, so anything that stops
     it from lifting strands the visitor on a frozen splash screen. Driving the
     dismissal only from a GSAP onComplete makes that outcome reachable: GSAP
     runs on requestAnimationFrame, which browsers pause outright in background
     tabs, so a page opened in a background tab could finish its boot log and
     then never uncover itself. This timer dismisses on wall-clock time no
     matter what the ticker is doing, and dismiss() is idempotent, so whichever
     path arrives first wins and the other becomes a no-op. */
  setTimeout(dismiss, holdFor + 1600);
}

/* ------------------------------------------------------------------ */
/* GSAP timelines                                                      */
/* ------------------------------------------------------------------ */
function animate() {
  gsap.registerPlugin(ScrollTrigger);
  if (HAS_MP) gsap.registerPlugin(MotionPathPlugin);
  gsap.defaults({ ease: 'power3.out' });

  // Distances collapse in the reduced tier; opacity and timing survive.
  const D = (px) => (FULL ? px : Math.sign(px) * Math.min(Math.abs(px), 8));
  const enterY = D(30);

  /* --- Hero entrance ------------------------------------------------ */
  // Built paused and returned to the caller, which plays it once the boot
  // overlay is out of the way. Building it now rather than after the boot
  // means the hero's opening state is set before anything is visible.
  const intro = gsap.timeline({ paused: true });
  intro
    .from('.hero-eyebrow', { opacity: 0, x: D(-18), duration: 0.7 })
    .from('#hero .wordmark .ch', {
      yPercent: FULL ? 110 : 0,
      opacity: FULL ? 1 : 0,
      duration: FULL ? 1.05 : 0.5,
      stagger: FULL ? 0.06 : 0.04,
      ease: 'expo.out',
    }, '-=0.35')
    .from('#heroLede', { opacity: 0, y: enterY, duration: 0.8 }, '-=0.55')
    .from('#heroActions .btn', { opacity: 0, y: D(16), duration: 0.6, stagger: 0.08 }, '-=0.45')
    .from('#scrollCue', { opacity: 0, duration: 0.6 }, '-=0.3');

  if (FULL) {
    intro.from('#adi', { opacity: 0, scale: 0.85, duration: 0.8, ease: 'back.out(1.6)' }, '-=0.7');
    gsap.to('.cue-dot', { y: 46, duration: 1.5, repeat: -1, ease: 'power1.inOut', opacity: 0 });

    // Hero recedes as the page scrolls past it.
    gsap.to('#heroInner', {
      y: -90, opacity: 0, ease: 'none',
      scrollTrigger: { trigger: '#hero', start: 'top top', end: 'bottom top', scrub: 0.4 },
    });
    gsap.to('.horizon-glow', {
      y: -160, ease: 'none',
      scrollTrigger: { trigger: '#hero', start: 'top top', end: 'bottom top', scrub: 0.6 },
    });
    gsap.to('.grid-floor', {
      backgroundPositionY: '46px', ease: 'none', duration: 2.4, repeat: -1,
    });
  }

  /* --- Section tags, with a decode on the label --------------------- */
  gsap.utils.toArray('.sec-tag').forEach((tag) => {
    gsap.from(tag, {
      opacity: 0, x: D(-14), duration: 0.6,
      scrollTrigger: {
        trigger: tag,
        start: 'top 88%',
        once: true,
        onEnter: () => {
          const s = tag.querySelector('.scramble');
          if (s && FULL) scramble(s, 700);
        },
      },
    });
  });

  /* --- The rule: words land one at a time --------------------------- */
  splitWords($('ruleCopy'));
  gsap.from('#ruleCopy .w', {
    opacity: 0,
    yPercent: FULL ? 60 : 0,
    rotateX: FULL ? -55 : 0,
    duration: FULL ? 0.75 : 0.45,
    stagger: FULL ? 0.035 : 0.02,
    scrollTrigger: { trigger: '#ruleCopy', start: 'top 80%' },
  });
  gsap.from('#refusalCard', {
    opacity: 0, y: enterY, duration: 0.8,
    scrollTrigger: { trigger: '#refusalCard', start: 'top 85%' },
  });
  if (FULL) {
    gsap.from('.refusal-bar', {
      scaleY: 0, transformOrigin: 'top', duration: 0.9, ease: 'power2.inOut',
      scrollTrigger: { trigger: '#refusalCard', start: 'top 85%' },
    });
  }

  /* --- Flight plan: route draws, aircraft flies it ------------------ */
  const path = $('routePath');
  const plane = $('aircraft');
  if (path) {
    const len = path.getTotalLength();
    gsap.set(path, { strokeDasharray: len, strokeDashoffset: len });
    const waypoints = gsap.utils.toArray('.wp');
    const litUpTo = (progress) => {
      waypoints.forEach((wp, i) => {
        wp.classList.toggle('lit', progress >= (i / waypoints.length) + 0.02);
      });
    };

    if (FULL && HAS_MP) {
      gsap.set(plane, { opacity: 0 });
      // Pinned so the route is drawn while the section is held, rather than
      // scrubbing past it — the whole flight is legible in one place.
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: '.pipeline',
          start: 'top top',
          end: '+=1400',
          pin: '#pipeSticky',
          scrub: 0.8,
          anticipatePin: 1,
          onUpdate: (self) => litUpTo(self.progress),
        },
      });
      tl.to(path, { strokeDashoffset: 0, ease: 'none', duration: 1 }, 0)
        .to(plane, { opacity: 1, duration: 0.04 }, 0)
        .to(plane, {
          motionPath: { path: path, align: path, alignOrigin: [0.5, 0.5], autoRotate: 90 },
          ease: 'none', duration: 1,
        }, 0)
        .to(plane, { opacity: 0, duration: 0.04 }, 0.97);
    } else {
      // Reduced tier: the route still draws (it explains the diagram), but on
      // its own clock, unpinned, with no aircraft chasing the scrollbar.
      gsap.set(plane, { opacity: 0 });
      gsap.to(path, {
        strokeDashoffset: 0, duration: 1.6, ease: 'power2.inOut',
        scrollTrigger: { trigger: '.route', start: 'top 82%', once: true },
        onUpdate: function () { litUpTo(this.progress()); },
        onComplete: () => litUpTo(1),
      });
    }
  }

  gsap.from('.stg', {
    opacity: 0, y: D(34), duration: 0.7, stagger: FULL ? 0.09 : 0.05,
    scrollTrigger: { trigger: '.stages-grid', start: 'top 88%' },
  });

  /* --- Instrument dials --------------------------------------------- */
  const CIRC = 2 * Math.PI * 50;
  gsap.utils.toArray('.gauge-cell').forEach((cell, i) => {
    const target = parseFloat(cell.dataset.val) || 0;
    const arc = cell.querySelector('.dial-arc');
    const num = cell.querySelector('.num');
    gsap.set(arc, { strokeDasharray: CIRC, strokeDashoffset: CIRC });

    const counter = { v: 0 };
    gsap.timeline({ scrollTrigger: { trigger: '.gauge-row', start: 'top 82%' }, delay: i * 0.11 })
      .to(arc, { strokeDashoffset: CIRC * (1 - target / 100), duration: 1.5, ease: 'power2.inOut' }, 0)
      .to(counter, {
        v: target, duration: 1.5, ease: 'power2.inOut',
        onUpdate: () => { num.textContent = counter.v.toFixed(target % 1 === 0 ? 0 : 1); },
      }, 0);
  });
  gsap.from('.gauge-cell', {
    opacity: 0, y: D(26), duration: 0.7, stagger: 0.08,
    scrollTrigger: { trigger: '.gauge-row', start: 'top 88%' },
  });

  /* --- Comparison bars ---------------------------------------------- */
  gsap.utils.toArray('.cmp-row').forEach((row) => {
    gsap.timeline({ scrollTrigger: { trigger: row, start: 'top 88%' } })
      .from(row, { opacity: 0, x: D(-16), duration: 0.5 }, 0)
      .to(row.querySelector('.bar.a i'), { width: row.dataset.a + '%', duration: 0.9, ease: 'power2.out' }, 0.1)
      .to(row.querySelector('.bar.b i'), { width: row.dataset.b + '%', duration: 1.1, ease: 'power2.out' }, 0.2);
  });

  /* --- Corpus marquee ------------------------------------------------ */
  const track = $('mqTrack');
  if (track && FULL) {
    track.innerHTML += track.innerHTML;
    const half = track.scrollWidth / 2;
    const drift = gsap.to(track, {
      x: -half, duration: 26, ease: 'none', repeat: -1,
      modifiers: { x: (x) => (parseFloat(x) % half) + 'px' },
    });
    ScrollTrigger.create({
      trigger: '.corpus', start: 'top bottom', end: 'bottom top',
      onUpdate: (self) => {
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
      scrollTrigger: { trigger: el, start: 'top 92%' },
      onUpdate: () => { el.textContent = Math.round(obj.v).toLocaleString('en-US'); },
    });
  });
  gsap.from('.corpus-stats div', {
    opacity: 0, y: D(24), duration: 0.6, stagger: 0.07,
    scrollTrigger: { trigger: '.corpus-stats', start: 'top 90%' },
  });

  /* --- CTA ----------------------------------------------------------- */
  if (FULL) {
    gsap.to('#sweep', { rotate: 360, duration: 5.5, repeat: -1, ease: 'none' });
    gsap.from('.cta-radar span', {
      scale: 0.7, opacity: 0, duration: 1.1, stagger: 0.12, ease: 'power2.out',
      scrollTrigger: { trigger: '.cta', start: 'top 78%' },
    });
  }
  gsap.from('.cta-h', {
    opacity: 0, y: D(40), duration: 0.9,
    scrollTrigger: { trigger: '.cta', start: 'top 75%' },
  });
  gsap.from('.cta-p, .cta .btn', {
    opacity: 0, y: D(20), duration: 0.7, stagger: 0.12,
    scrollTrigger: { trigger: '.cta', start: 'top 72%' },
  });

  /* Webfonts swap after first paint and change every element's height, which
     leaves every ScrollTrigger measuring stale positions. This is the single
     most common cause of triggers firing at the wrong scroll offset. */
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => ScrollTrigger.refresh());
  }
  window.addEventListener('load', () => ScrollTrigger.refresh());

  return intro;
}

/* ------------------------------------------------------------------ */
/* Anchor scrolling — replaces CSS scroll-behavior, which breaks scrub */
/* ------------------------------------------------------------------ */
function wireAnchors() {
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', (e) => {
      const target = document.querySelector(a.getAttribute('href'));
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: FULL ? 'smooth' : 'auto', block: 'start' });
    });
  });
}

/* ------------------------------------------------------------------ */
document.documentElement.dataset.motion = MOTION;
wireMotionToggle();
buildHeadingTape();
wireHudToScroll();
wireAnchors();
startSky();
loadHealth();

if (HAS_GSAP && HAS_ST) {
  wirePointerFlourishes();
  // animate() runs now, not after the boot: every ScrollTrigger and opening
  // state is established up front, so a stalled boot can never leave the page
  // un-animated. The boot only decides when the hero timeline starts playing.
  const intro = animate();
  runBoot(() => {
    intro.play();
    // Same reasoning as the boot watchdog: the hero's opening state is
    // opacity 0, so if the ticker never advances the headline would simply
    // never appear. If the timeline has not moved shortly after being told
    // to play, snap it to its finished state.
    setTimeout(() => { if (intro.progress() === 0) intro.progress(1); }, 4000);
  });
} else {
  // No GSAP at all: paint every final state and drop the boot cover.
  applyStaticFallback();
  const boot = $('boot');
  if (boot) boot.classList.add('done');
  document.body.style.overflow = '';
}
