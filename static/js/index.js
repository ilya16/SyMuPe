window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';

            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');

    if (carouselVideos.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });

    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    bulmaSlider.attach();

    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();
})

// =====================================================
// ====== CUSTOM AUDIO PLAYER LOGIC (V4 - Robust) ======
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
    const players = document.querySelectorAll('.custom-player');
    const allAudioElements = document.querySelectorAll('.custom-player audio');

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
    }

    players.forEach(player => {
        const audio = player.querySelector('audio');
        const playPauseBtn = player.querySelector('.play-pause-btn'); // Get the button
        const progressBar = player.querySelector('.progress-bar');
        const progress = player.querySelector('.progress');
        const timeDisplay = player.querySelector('.time-display');

        function togglePlay(e) {
            // Stop the click from bubbling up to the player div, just in case.
            e.stopPropagation();

            if (audio.paused) {
                // Pause all others
                allAudioElements.forEach(otherAudio => {
                    if (otherAudio !== audio) otherAudio.pause();
                });
                audio.play();
            } else {
                audio.pause();
            }
        }

        // Attach the listener to the BUTTON, not the 'player' div
        playPauseBtn.addEventListener('click', togglePlay);

        // Add/remove .is-playing class on the container
        audio.addEventListener('play', () => player.classList.add('is-playing'));
        audio.addEventListener('pause', () => player.classList.remove('is-playing'));
        audio.addEventListener('ended', () => player.classList.remove('is-playing'));

        audio.addEventListener('loadedmetadata', () => {
            if (audio.duration) {
                timeDisplay.textContent = `0:00 / ${formatTime(audio.duration)}`;
            }
        });

        audio.addEventListener('timeupdate', () => {
            if (audio.duration) {
                progress.style.width = `${(audio.currentTime / audio.duration) * 100}%`;
                timeDisplay.textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
            }
        });

        audio.addEventListener('ended', () => {
            progress.style.width = '0%';
            timeDisplay.textContent = `0:00 / ${formatTime(audio.duration)}`;
        });

        progressBar.addEventListener('click', (e) => {
            e.stopPropagation();
            const width = progressBar.clientWidth;
            const clickX = e.offsetX;
            const duration = audio.duration;
            if (duration) audio.currentTime = (clickX / width) * duration;
        });
    });
});

// =====================================================
// ====== NAVBAR: ACTIVE LINK & MOBILE BURGER LOGIC (V3) ======
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const burger = document.querySelector('.navbar-burger');
    const menu = document.querySelector('.navbar-menu');
    const brandLink = document.querySelector('.navbar-brand-link'); // The "SyMuPe" link
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.navbar-menu a.navbar-item');

    // --- Mobile Burger Menu Logic ---
    burger.addEventListener('click', () => {
        burger.classList.toggle('is-active');
        menu.classList.toggle('is-active');
    });

    // --- Universal function to set the active link ---
    function setActiveLink(id) {
        // Deactivate brand link first, by default
        brandLink.classList.remove('is-active');
        // Loop through section links
        navLinks.forEach(link => {
            if (link.getAttribute('href') === `#${id}`) {
                link.classList.add('is-active');
            } else {
                link.classList.remove('is-active');
            }
        });
    }

    // --- IntersectionObserver for efficient section detection ---
    const observerOptions = {
        root: null,
        rootMargin: '-40% 0px -40% 0px',
        threshold: 0
    };

    const observer = new IntersectionObserver((entries, observer) => {
        // Check if we are at the top of the page; if so, do nothing and let the scroll handler take over.
        if (window.scrollY < 100) return;

        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                setActiveLink(id);
            }
        });
    }, observerOptions);

    sections.forEach(section => {
        observer.observe(section);
    });

    // --- Scroll Listener for Edge Cases (Top and Bottom of page) ---
    window.addEventListener('scroll', () => {
        const atTop = window.scrollY < 100;
        const atBottom = (window.innerHeight + window.scrollY) >= document.body.offsetHeight - 50;

        if (atTop) {
            // When at the very top, force the brand link to be active
            navLinks.forEach(link => link.classList.remove('is-active'));
            // brandLink.classList.add('is-active');
        } else if (atBottom) {
            // When at the very bottom, force the last link to be active
            const lastSectionId = sections[sections.length - 1].getAttribute('id');
            setActiveLink(lastSectionId);
        }
        // In the middle, the IntersectionObserver handles everything.
    });

    // --- Initial Check on Page Load ---
    // Manually trigger the scroll handler logic once to set the correct initial state.
    const initialScrollEvent = new Event('scroll');
    window.dispatchEvent(initialScrollEvent);
});

// =====================================================
// ====== AUDIO LAZY LOADING LOGIC               ======
// =====================================================
document.addEventListener("DOMContentLoaded", () => {
  const mainVideo = document.getElementById("teaser-video");
  const delayedAudios = document.querySelectorAll("audio");

  if (!mainVideo) {
    return;
  }

  mainVideo.addEventListener("canplaythrough", () => {
    delayedAudios.forEach(audio => {
      audio.load();
    });
  }, { once: true });
});