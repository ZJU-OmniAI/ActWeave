(() => {
  'use strict';
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = matchMedia('(hover: hover) and (pointer: fine)');
  const phaseCards = [...document.querySelectorAll('[data-phase]')];
  const colors = ['var(--intent)', 'var(--program)', 'var(--evidence)'];
  const phaseWeave = document.getElementById('phase-weave');
  let phase = 0;
  let angle = 0;
  let animationFrame;
  function weave(svg, offset) {
    const length = svg.clientWidth * 110 / svg.clientHeight;
    svg.setAttribute('viewBox', `0 0 ${length} 110`);
    offset %= Math.PI * 2;
    const amplitude = 25;
    const period = 180;
    const width = 4.5;
    const shifts = colors.map((_, index) => index * Math.PI * 2 / 3);
    const point = (position, index) => {
      const wave = Math.sin(position / period * Math.PI * 2 + offset + shifts[index]) * amplitude;
      return `${position},${55 + wave}`;
    };
    const path = (from, to, index, samples) => Array.from({length:samples + 1}, (_, step) => `${step ? 'L' : 'M'}${point(from + (to - from) * step / samples, index)}`).join(' ');
    const opacity = index => index === phase ? 1 : .46;
    const cuts = colors.map(() => []);
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        for (let turn = -4; turn < Math.ceil(length * 2 / period) + 6; turn++) {
          const position = (Math.PI / 2 - (shifts[a] + shifts[b]) / 2 - offset + turn * Math.PI) * period / (Math.PI * 2);
          if (position < -8 || position > length + 8) continue;
          const angleAtCrossing = position / period * Math.PI * 2 + offset;
          const front = Math.cos(angleAtCrossing + shifts[a]) > Math.cos(angleAtCrossing + shifts[b]) ? a : b;
          const back = front === a ? b : a;
          const crossing = path(position - 4, position + 4, front, 8);
          cuts[back].push(`<path d="${crossing}" fill="none" stroke="black" stroke-width="${width * 2.15}" stroke-linecap="round"/>`);
        }
      }
    }
    const masks = cuts.map((segments, index) => `<mask id="${svg.id}-strand-${index}" maskUnits="userSpaceOnUse" x="0" y="0" width="${length}" height="110"><rect width="${length}" height="110" fill="white"/>${segments.join('')}</mask>`).join('');
    svg.innerHTML = `<defs>${masks}</defs>` + colors.map((color, index) => `<path d="${path(-width,length + width,index,Math.ceil(length / 3))}" fill="none" stroke="${color}" stroke-width="${width}" stroke-opacity="${opacity(index)}" stroke-linecap="round" mask="url(#${svg.id}-strand-${index})"/>`).join('');
  }
  function setPhase(index) {
    phase = index;
    phaseCards.forEach((card, i) => {
      card.setAttribute('aria-pressed', String(i === index));
    });
    cancelAnimationFrame(animationFrame);
    const startAngle = angle;
    const fullTurn = Math.PI * 2;
    const targetPhase = index * fullTurn / 3;
    const delta = ((targetPhase - angle) % fullTurn + fullTurn) % fullTurn;
    const target = angle + (delta < .01 ? fullTurn : delta);
    if (reducedMotion.matches) { angle = targetPhase; weave(phaseWeave, angle); return; }
    const started = performance.now();
    const animate = now => {
      const progress = Math.min(1, (now - started) / 1100);
      angle = startAngle + (target - startAngle) * (1 - Math.pow(1 - progress, 3));
      weave(phaseWeave, angle);
      if (progress < 1) animationFrame = requestAnimationFrame(animate);
      else angle %= fullTurn;
    };
    animationFrame = requestAnimationFrame(animate);
  }
  phaseCards.forEach((card, index) => {
    card.addEventListener('pointerenter', event => { if (event.pointerType === 'mouse') setPhase(index); });
    card.addEventListener('focus', () => setPhase(index));
    card.addEventListener('click', () => setPhase(index));
    card.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setPhase(index); }
    });
  });
  weave(phaseWeave, angle);
  new ResizeObserver(() => weave(phaseWeave, angle)).observe(phaseWeave);
  function updateTilt() {
    document.querySelectorAll('.tilt-surface').forEach(element => {
      if (reducedMotion.matches || !finePointer.matches) {
        if (element.vanillaTilt) element.vanillaTilt.destroy();
      } else if (!element.vanillaTilt) {
        VanillaTilt.init(element, {max:6, perspective:1100, scale:1.015, speed:450, glare:false, gyroscope:false});
      }
    });
  }
  updateTilt();
  reducedMotion.addEventListener('change', updateTilt);
  finePointer.addEventListener('change', updateTilt);
  const technologies = [
    {name:'GPT-6 Astra',role:'Planning and recovery',icon:'intent',color:'intent',description:'Interprets the task and current evidence, chooses a manipulation method, and writes executable programs. It decides when to observe, refine a target, or change the operation.'},
    {name:'Effect Thread',role:'Operation structure',icon:'link',color:'intent',description:'Gives a physical operation a persistent identity and revision. Its intent, assumptions, expected effects, and preserved relations remain linked to execution and observation across decisions.'},
    {name:'Persistent Python',role:'Program execution',icon:'code',color:'intent',description:'Runs geometric computations, robot actions, and numerical-feedback branches in persistent cells. Variables and helpers carry forward so programs can build on earlier measurements and execution.'},
    {name:'SAM3',role:'Object segmentation',icon:'camera',color:'intent',description:'Selects objects in public RGB observations using text, points, or bounding boxes. The resulting masks provide image evidence for geometric measurement in the current scene.'},
    {name:'Spatial Geometry',role:'Geometric grounding',icon:'grid',color:'intent',description:'Uses camera calibration, image masks, and multiple views to measure object geometry and functional features. Fits, rays, and residuals retain their source observations.'},
    {name:'Hand-Object Bindings',role:'Object-centered targets',icon:'link',color:'intent',description:'Connects measured object features to the gripper frame. Object-space goals become candidate robot poses, while new measurements can be compared with the bound geometric prediction.'},
    {name:'Motion Control',role:'Planning and execution',icon:'intent',color:'intent',description:'Combines calibrated kinematics, inverse-kinematics solutions, Cartesian or joint paths, and gripper commands. Actual endpoints, progress, and errors return to the calling program.'},
    {name:'Execution Evidence',role:'Receipts and observations',icon:'layers',color:'intent',description:'Associates measurements and physical calls with their programs and operations. Original frames, execution intervals, errors, and relevant history form the evidence for the next decision.'},
    {name:'Jev',role:'Visual interpretation',icon:'eye',color:'intent',description:'Answers visual questions using selected original images and operation evidence. Its judgments return to Astra, alongside numerical feedback, to inform effect reconciliation.'},
    {name:'Demonstrations & Experience',role:'Method references',icon:'camera',color:'intent',description:'Provides curated training-video references and recorded manipulation guidance. Astra can inspect contact regions, hand roles, and operation stages, then adapt the method to the current scene.'},
    {name:'DeepSeek Harness',role:'Agent runtime',icon:'layers',color:'intent',description:'Connects model calls, tools, persistent sessions, attachments, and Python execution. It carries programs and their returned evidence through the online Agent loop.'}
  ];
  document.getElementById('technology-cards').innerHTML = [...technologies, ...technologies].map((item, position) => `<article class="swiper-slide tech-card" data-index="${position % technologies.length}" style="--tech-color:var(--${item.color})"><span class="tech-category">${item.role}</span><div class="tech-symbol"><svg aria-hidden="true"><use href="#i-${item.icon}"/></svg></div><h3>${item.name}</h3><p>${item.description}</p></article>`).join('');
  const carousel = new Swiper('#technology-carousel', {
    effect:'coverflow', centeredSlides:true, slidesPerView:'auto', loop:true, initialSlide:technologies.length + 1,
    speed:reducedMotion.matches ? 0 : 650, grabCursor:true, loopAdditionalSlides:1, loopPreventsSliding:false,
    coverflowEffect:{rotate:24, stretch:55, depth:180, scale:.9, slideShadows:false},
    keyboard:{enabled:true, onlyInViewport:true},
    a11y:{slideLabelMessage:''},
    on:{setTranslate(swiper) {
      for (const slide of swiper.slides) {
        const visible = Math.abs(slide.progress) < 3.5;
        slide.style.opacity = visible ? '1' : '0';
        slide.style.pointerEvents = visible ? 'auto' : 'none';
        slide.setAttribute('aria-hidden', String(slide !== swiper.slides[swiper.activeIndex]));
      }
    }}
  });
  carousel.slides.forEach(slide => slide.addEventListener('click', () => {
    if (!carousel.allowClick) return;
    const offset = carousel.slides.indexOf(slide) - carousel.activeIndex;
    if (!offset) return;
    carousel.loopFix({direction:offset < 0 ? 'prev' : 'next'});
    carousel.wrapperEl.getBoundingClientRect();
    carousel.slideTo(carousel.slides.indexOf(slide));
  }));
  const cases = [
    {stem:'nesting-dolls',title:'Nesting dolls',description:'Arrange the dolls in size order. Successive grasps and placements must establish the new ordering while preserving the positions of dolls already arranged.'},
    {stem:'stack-bowls',title:'Stack bowls',description:'Acquire each bowl at its rim, transfer it over the stack, then release and withdraw. Retention during transport gives way to stable support after placement.'},
    {stem:'build-tower',title:'Build a tower',description:'Build a supported structure one tier at a time. Position the supports, bridge them with boards, and place the top pieces while preserving the structure beneath.'},
    {stem:'swap-t',title:'Swap T-shaped blocks',description:'Exchange the two blocks using both arms. Preserve each target orientation and coordinate the transfers so the blocks reach their opposite destinations.'},
    {stem:'cover-blocks',title:'Cover and uncover blocks',description:'Cover the blocks from left to right while retaining their color identities, then uncover them in the requested red, green, and blue order.'},
    {stem:'align-blocks',title:'Align blocks',description:'Use a set square to push three blocks into a straight row. The tool must establish the object alignment before the arm returns to its initial pose.'},
    {stem:'pour-liquid',title:'Pour liquid',description:'Grasp the bottle, bring its opening over the cup, and tilt it to transfer the liquid. The operation connects the bottle pose to the changing physical effect at the cup.'},
    {stem:'play-xylophone',title:'Play the xylophone',description:'Pick up the mallet and strike the keys from left to right. Tool acquisition and ordered contacts form a sequence of physical operations.'}
  ];
  const deck = document.getElementById('case-deck');
  deck.innerHTML = cases.map(item => `<li role="button" aria-label="${item.title}: next case" tabindex="-1"><article class="case-paper"><img src="assets/media/${item.stem}-poster.jpg" alt="${item.title} in RoboDojo" draggable="false" width="640" height="480"><div class="case-paper-bottom"><h4>${item.title}</h4></div></article></li>`).join('');
  let caseIndex = 0;
  let caseVisible = false;
  let switching = false;
  function updateMedia() {
    [...deck.children].forEach((card,index) => {
      const active = index === caseIndex;
      const animated = active && !reducedMotion.matches && caseVisible && !document.hidden;
      const url = `assets/media/${cases[index].stem}${animated ? '.gif' : '-poster.jpg'}`;
      const image = card.querySelector('img');
      if (image.getAttribute('src') !== url) image.src = url;
      card.setAttribute('aria-hidden', String(!active));
      card.tabIndex = active ? 0 : -1;
      card.style.pointerEvents = active ? 'auto' : 'none';
    });
  }
  function updateCase(index) {
    switching = false;
    caseIndex = index;
    document.getElementById('case-description').textContent = cases[index].description;
    updateMedia();
    if (deck.contains(document.activeElement)) deck.children[index].focus({preventScroll:true});
  }
  const stack = new ElastiStack(deck, {distDragBack:100, distDragMax:260, onUpdateStack:updateCase});
  function nextCase() {
    if (switching || !stack.draggie.isEnabled) return;
    switching = true;
    stack.draggie.position.x = 150;
    stack.draggie.position.y = -25;
    stack._moveAway(stack.draggie);
  }
  let pressPosition;
  let dragged = false;
  deck.addEventListener('pointerdown', event => { pressPosition = [event.clientX, event.clientY]; dragged = false; });
  deck.addEventListener('pointermove', event => {
    if (pressPosition && Math.hypot(event.clientX - pressPosition[0], event.clientY - pressPosition[1]) > 6) dragged = true;
  });
  deck.addEventListener('click', () => { if (!dragged) nextCase(); });
  deck.addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    nextCase();
  });
  new IntersectionObserver(entries => {caseVisible = entries[0].isIntersecting; updateMedia();}, {threshold:.15}).observe(deck);
  document.addEventListener('visibilitychange', updateMedia);
  updateCase(0);
  const results = [
    {dimension:'Generalization',sr:[20.83,45.83],score:[23.75,49.58]},
    {dimension:'Precision',sr:[25.00,31.25],score:[32.81,40.78]},
    {dimension:'Long-Horizon',sr:[25.00,25.00],score:[42.81,49.22]},
    {dimension:'Memory',sr:[75.00,83.33],score:[76.25,83.33]},
    {dimension:'Open',sr:[59.38,53.13],score:[61.41,54.53]},
    {dimension:'Overall',sr:[41.04,47.71],score:[47.41,55.49]}
  ];
  const resultCharts = document.querySelectorAll('.result-chart');
  resultCharts.forEach(chart => {
    const metric = chart.dataset.metric;
    chart.querySelector('.result-rows').innerHTML = results.map((row,index) => `<div class="result-row${index === results.length - 1 ? ' result-overall' : ''}"><h4>${row.dimension}</h4><div class="result-bars">${row[metric].map((value,series) => `<div class="result-bar ${series ? 'actweave-bar' : 'baseline-bar'}" style="--value:${value}%;--delay:${index * 65 + series * 45}ms" role="img" aria-label="${row.dimension}, ${series ? 'ActWeave' : 'Baseline'} ${metric === 'sr' ? 'SR' : 'Score'}: ${value.toFixed(2)}${metric === 'sr' ? '%' : ''}"><span class="result-fill" aria-hidden="true"></span><span class="result-value" aria-hidden="true">${value.toFixed(2)}</span></div>`).join('')}</div></div>`).join('');
  });
  const resultObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.intersectionRatio >= .25) entry.target.classList.add('is-revealed');
      else if (!entry.isIntersecting) entry.target.classList.remove('is-revealed');
    });
  }, {threshold:[0,.25]});
  resultCharts.forEach(chart => resultObserver.observe(chart));
  const navigation = [...document.querySelectorAll('.rail-link')];
  const sections = navigation.map(link => document.querySelector(link.getAttribute('href')));
  let scrollPending = false;
  function updateNavigation() {
    let active = 0;
    sections.forEach((section,index) => {if(section.getBoundingClientRect().top < innerHeight * .35) active = index;});
    if(scrollY + innerHeight >= document.documentElement.scrollHeight - 8) active = sections.length - 1;
    navigation.forEach((link,index) => {
      link.classList.toggle('active',index === active);
      if(index === active) link.setAttribute('aria-current','location'); else link.removeAttribute('aria-current');
    });
    scrollPending = false;
  }
  addEventListener('scroll', () => {if(!scrollPending){scrollPending = true; requestAnimationFrame(updateNavigation);}}, {passive:true});
  updateNavigation();
})();
