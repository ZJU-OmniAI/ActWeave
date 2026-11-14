# Third-party components

## Swiper 14.4.0

- Source: https://github.com/nolimits4web/swiper/tree/v14.4.0
- Official example: https://github.com/nolimits4web/swiper/blob/v14.4.0/demos/240-effect-coverflow.html
- Browser distribution: https://registry.npmjs.org/swiper/-/swiper-14.4.0.tgz
- License: MIT; retained in `assets/vendor/swiper/LICENSE` and distribution headers.
- Adaptation: Coverflow configuration, ActWeave content, theme colors and navigation. The distributed library is unmodified.

## Vanilla Tilt

- Source: https://github.com/micku7zu/vanilla-tilt.js
- Distribution: `dist/vanilla-tilt.min.js`, downloaded from the upstream repository.
- License: MIT; retained in `assets/vendor/vanilla-tilt/LICENSE`.
- The original library supplies pointer-tracked 3D tilt for the two figures and three method cards. ActWeave adds hover elevation and stage selection; reduced-motion and touch-only devices retain static cards.

## ElastiStack

- Source: https://github.com/codrops/ElastiStack
- Original component: `js/elastiStack.js`, `css/component.css`.
- License: MIT, Copyright 2013 Codrops; original license headers retained.
- Bundled upstream dependencies: Draggabilly and Modernizr, retaining their license headers. Classie is from https://github.com/desandro/classie (MIT).
- Adaptation: the original ElastiStack drag, elastic following, release and reorder implementation is used unchanged. Card content, surfaces and fixed paper rotations are customized. Card clicks call the original transition; the stack callback loads an animated GIF only into the active, visible card. Background cards use still posters.

## Day-night toggle

- Source: https://github.com/Xiumuzaidiao/Day-night-toggle-button
- License: ISC; retained in `assets/vendor/day-night/LICENSE`.
- The existing Oh My Evo v1 CSS adaptation is reused with ActWeave theme state and accessible control labels.

## Fonts

Sora, Manrope and DM Mono are obtained from https://github.com/google/fonts/tree/main/ofl and locally hosted. Their SIL Open Font License files are retained beside each font in `assets/fonts/`.

## Simulation media

- The eight case GIFs use the original head-camera frame sequences of successful ActWeave Agent episodes: nesting-doll sorting, bowl stacking, tower building, swapping T-shaped blocks, covering blocks, aligning blocks, pouring liquid, and playing the xylophone.
- The displayed sequences cover each recorded episode from its first frame through native completion. Playback is accelerated, with 12-frame-per-second GIF output and a final-frame hold.
- Posters are the first frames of their corresponding sequences. Only the active, visible task card loads its GIF.
