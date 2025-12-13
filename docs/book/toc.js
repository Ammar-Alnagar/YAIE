// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="intro/welcome.html"><strong aria-hidden="true">1.</strong> Introduction</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="intro/prerequisites.html"><strong aria-hidden="true">1.1.</strong> Prerequisites</a></li><li class="chapter-item expanded "><a href="intro/setup.html"><strong aria-hidden="true">1.2.</strong> Environment Setup</a></li></ol></li><li class="chapter-item expanded "><a href="concepts/llm_inference.html"><strong aria-hidden="true">2.</strong> Core Concepts</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="concepts/continuous_batching.html"><strong aria-hidden="true">2.1.</strong> Continuous Batching</a></li><li class="chapter-item expanded "><a href="concepts/radix_attention.html"><strong aria-hidden="true">2.2.</strong> Radix Attention (SGLang)</a></li><li class="chapter-item expanded "><a href="concepts/paged_attention.html"><strong aria-hidden="true">2.3.</strong> Paged Attention (vLLM)</a></li></ol></li><li class="chapter-item expanded "><a href="architecture/system_overview.html"><strong aria-hidden="true">3.</strong> Architecture Deep Dive</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="architecture/engine.html"><strong aria-hidden="true">3.1.</strong> Engine Orchestration</a></li><li class="chapter-item expanded "><a href="architecture/scheduler.html"><strong aria-hidden="true">3.2.</strong> Scheduler Logic</a></li><li class="chapter-item expanded "><a href="architecture/memory_manager.html"><strong aria-hidden="true">3.3.</strong> Memory Management</a></li></ol></li><li class="chapter-item expanded "><a href="kernels/python/overview.html"><strong aria-hidden="true">4.</strong> Python Kernels Guide</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="kernels/python/radix_tree.html"><strong aria-hidden="true">4.1.</strong> Radix Tree (Trie)</a></li><li class="chapter-item expanded "><a href="kernels/python/kv_cache_manager.html"><strong aria-hidden="true">4.2.</strong> KV Cache Manager</a></li><li class="chapter-item expanded "><a href="kernels/python/radix_attention_module.html"><strong aria-hidden="true">4.3.</strong> Radix Attention Module</a></li><li class="chapter-item expanded "><a href="kernels/python/sampling_module.html"><strong aria-hidden="true">4.4.</strong> Sampling Logic</a></li></ol></li><li class="chapter-item expanded "><a href="kernels/cuda/setup.html"><strong aria-hidden="true">5.</strong> CUDA Kernels Guide</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="kernels/cuda/memory_ops.html"><strong aria-hidden="true">5.1.</strong> Memory Operations</a></li><li class="chapter-item expanded "><a href="kernels/cuda/flash_attention.html"><strong aria-hidden="true">5.2.</strong> Flash Attention</a></li><li class="chapter-item expanded "><a href="kernels/cuda/paged_attention.html"><strong aria-hidden="true">5.3.</strong> Paged Attention</a></li><li class="chapter-item expanded "><a href="kernels/cuda/radix_ops.html"><strong aria-hidden="true">5.4.</strong> Radix Operations</a></li></ol></li><li class="chapter-item expanded "><a href="serving/api_endpoints.html"><strong aria-hidden="true">6.</strong> API &amp; Serving</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="serving/cli.html"><strong aria-hidden="true">6.1.</strong> CLI Usage</a></li><li class="chapter-item expanded "><a href="serving/production.html"><strong aria-hidden="true">6.2.</strong> Production Deployment</a></li></ol></li><li class="chapter-item expanded "><a href="appendix/references.html"><strong aria-hidden="true">7.</strong> Appendices</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="appendix/troubleshooting.html"><strong aria-hidden="true">7.1.</strong> Troubleshooting</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
