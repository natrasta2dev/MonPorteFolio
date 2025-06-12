// ========== Animation à l’apparition ==========
const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('visible');
    });
});
document.querySelectorAll('section').forEach(section => observer.observe(section));

// ========== Carrousel infini ==========
const track = document.querySelector('.carousel-track');
if (track) track.innerHTML += track.innerHTML;
