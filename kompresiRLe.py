import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import threading
import os
import time
import json
from math import log10, sqrt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import webbrowser
from datetime import datetime
import platform
import psutil
import sys

class DCTCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum DCT Image Compressor")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#f0f2f5")
        
        # Application state variables
        self.filename = None
        self.original_image = None
        self.processed_image = None
        self.dct_coefficients = None
        self.compression_ratio = 1.0
        self.psnr_value = 0.0
        self.file_size_before = 0
        self.file_size_after = 0
        self.history = []
        self.settings = self.load_settings()
        
        # UI styling
        self.style = {
            'font': ('Segoe UI', 10),
            'bg': '#f0f2f5',
            'fg': '#333',
            'activebg': '#3a7ca5',
            'activefg': 'white',
            'highlight': '#2f6690',
            'accent': '#3a7ca5',
            'dark': '#16425b',
            'light': '#d9dcd6',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336'
        }
        
        # Initialize UI
        self.setup_ui()
        self.create_menu()
        self.create_toolbar()
        self.create_main_panels()
        self.create_status_bar()
        
        # Load default settings
        self.load_default_settings()
        
        # System monitoring
        self.start_system_monitor()

    def setup_ui(self):
        """Configure the main window appearance"""
        self.root.iconbitmap(self.resource_path('icon.ico')) if os.path.exists(self.resource_path('icon.ico')) else None
        self.root.option_add('*tearOff', False)
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=self.style['bg'])
        style.configure('TLabel', background=self.style['bg'], foreground=self.style['fg'])
        style.configure('TButton', 
                       background=self.style['accent'], 
                       foreground='white',
                       borderwidth=1,
                       font=self.style['font'])
        style.map('TButton',
                 background=[('active', self.style['highlight']), 
                            ('pressed', self.style['dark'])])
        style.configure('TEntry', fieldbackground='white')
        style.configure('TCombobox', fieldbackground='white')
        style.configure('TProgressbar', 
                        background=self.style['accent'],
                        troughcolor=self.style['light'],
                        thickness=10)
        style.configure('Vertical.TScrollbar', 
                       background=self.style['light'],
                       troughcolor=self.style['bg'],
                       bordercolor=self.style['bg'],
                       arrowcolor=self.style['fg'])
        
        # Custom styles
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 16, 'bold'),
                       foreground=self.style['dark'])
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 12),
                       foreground=self.style['highlight'])
        style.configure('Accent.TButton',
                       background=self.style['highlight'],
                       foreground='white',
                       font=('Segoe UI', 11, 'bold'))
        style.configure('Success.TButton',
                       background=self.style['success'],
                       foreground='white')
        style.configure('Warning.TButton',
                       background=self.style['warning'],
                       foreground='white')
        
    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
            
        return os.path.join(base_path, relative_path)

    def create_menu(self):
        """Create the main menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Processed Image", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Recent Files", command=self.show_recent_files)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_exit, accelerator="Alt+F4")
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo_action, accelerator="Ctrl+Z", state='disabled')
        edit_menu.add_command(label="Redo", command=self.redo_action, accelerator="Ctrl+Y", state='disabled')
        edit_menu.add_separator()
        edit_menu.add_command(label="Preferences", command=self.show_preferences)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # Process menu
        process_menu = tk.Menu(menubar, tearoff=0)
        process_menu.add_command(label="Apply DCT", command=self.apply_dct, accelerator="F5")
        process_menu.add_command(label="Compare Images", command=self.show_comparison)
        process_menu.add_separator()
        process_menu.add_command(label="Show DCT Coefficients", command=self.show_dct_coefficients)
        process_menu.add_command(label="Show Frequency Domain", command=self.show_frequency_domain)
        menubar.add_cascade(label="Process", menu=process_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.open_image())
        self.root.bind('<Control-s>', lambda e: self.save_image())
        self.root.bind('<F5>', lambda e: self.apply_dct())
        
    def create_toolbar(self):
        """Create the toolbar with quick access buttons"""
        toolbar = ttk.Frame(self.root, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # Toolbar buttons
        icons = {
            'open': '📂',
            'save': '💾',
            'dct': '🌀',
            'compare': '🔍',
            'undo': '↩️',
            'redo': '↪️',
            'settings': '⚙️',
            'help': '❓'
        }
        
        btn_open = ttk.Button(toolbar, text=f"{icons['open']} Open", command=self.open_image, style='Accent.TButton')
        btn_save = ttk.Button(toolbar, text=f"{icons['save']} Save", command=self.save_image, style='Accent.TButton')
        btn_dct = ttk.Button(toolbar, text=f"{icons['dct']} DCT", command=self.apply_dct, style='Success.TButton')
        btn_compare = ttk.Button(toolbar, text=f"{icons['compare']} Compare", command=self.show_comparison)
        btn_undo = ttk.Button(toolbar, text=f"{icons['undo']} Undo", command=self.undo_action, state='disabled')
        btn_redo = ttk.Button(toolbar, text=f"{icons['redo']} Redo", command=self.redo_action, state='disabled')
        btn_settings = ttk.Button(toolbar, text=f"{icons['settings']} Settings", command=self.show_preferences)
        btn_help = ttk.Button(toolbar, text=f"{icons['help']} Help", command=self.show_documentation)
        
        # Pack buttons
        btn_open.pack(side=tk.LEFT, padx=2, pady=2)
        btn_save.pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        btn_dct.pack(side=tk.LEFT, padx=2, pady=2)
        btn_compare.pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        btn_undo.pack(side=tk.LEFT, padx=2, pady=2)
        btn_redo.pack(side=tk.LEFT, padx=2, pady=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        btn_settings.pack(side=tk.LEFT, padx=2, pady=2)
        btn_help.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Store references for state management
        self.toolbar_buttons = {
            'save': btn_save,
            'dct': btn_dct,
            'compare': btn_compare,
            'undo': btn_undo,
            'redo': btn_redo
        }
        
    def create_main_panels(self):
        """Create the main content panels"""
        main_panel = ttk.Frame(self.root)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - controls and info
        left_panel = ttk.Frame(main_panel, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Right panel - image display
        right_panel = ttk.Frame(main_panel)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create left panel content
        self.create_controls_panel(left_panel)
        self.create_info_panel(left_panel)
        self.create_histogram_panel(left_panel)
        
        # Create right panel content
        self.create_image_display(right_panel)
        
    def create_controls_panel(self, parent):
        """Create the controls panel with DCT settings"""
        controls_frame = ttk.LabelFrame(parent, text="DCT Compression Settings", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Compression level
        ttk.Label(controls_frame, text="Compression Level:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.compression_level = tk.IntVar(value=50)
        self.slider_compression = ttk.Scale(controls_frame, from_=1, to=100, variable=self.compression_level,
                                          command=lambda e: self.update_compression_label())
        self.slider_compression.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 5))
        
        self.lbl_compression_value = ttk.Label(controls_frame, text="50%")
        self.lbl_compression_value.grid(row=1, column=2, padx=(5, 0))
        
        # Block size
        ttk.Label(controls_frame, text="Block Size:").grid(row=2, column=0, sticky=tk.W, pady=(5, 5))
        self.block_size = tk.StringVar(value="8x8")
        block_sizes = ["4x4", "8x8", "16x16", "32x32"]
        self.cmb_block_size = ttk.Combobox(controls_frame, textvariable=self.block_size, values=block_sizes, state="readonly")
        self.cmb_block_size.grid(row=2, column=1, sticky=tk.EW, pady=(5, 5))
        
        # Color space
        ttk.Label(controls_frame, text="Color Space:").grid(row=3, column=0, sticky=tk.W, pady=(5, 5))
        self.color_space = tk.StringVar(value="YCbCr")
        color_spaces = ["RGB", "YCbCr", "Grayscale"]
        self.cmb_color_space = ttk.Combobox(controls_frame, textvariable=self.color_space, values=color_spaces, state="readonly")
        self.cmb_color_space.grid(row=3, column=1, sticky=tk.EW, pady=(5, 5))
        
        # Quantization matrix
        ttk.Label(controls_frame, text="Quantization:").grid(row=4, column=0, sticky=tk.W, pady=(5, 5))
        self.quantization = tk.StringVar(value="Standard JPEG")
        quantizations = ["Standard JPEG", "Custom", "Uniform"]
        self.cmb_quantization = ttk.Combobox(controls_frame, textvariable=self.quantization, values=quantizations, state="readonly")
        self.cmb_quantization.grid(row=4, column=1, sticky=tk.EW, pady=(5, 5))
        
        # Process button
        btn_process = ttk.Button(controls_frame, text="Apply DCT Compression", command=self.apply_dct, style='Success.TButton')
        btn_process.grid(row=5, column=0, columnspan=3, pady=(10, 0), sticky=tk.EW)
        
        # Configure grid weights
        controls_frame.columnconfigure(1, weight=1)
        
    def create_info_panel(self, parent):
        """Create the information panel with image stats"""
        info_frame = ttk.LabelFrame(parent, text="Image Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Image info labels
        self.lbl_filename = ttk.Label(info_frame, text="File: None")
        self.lbl_filename.pack(anchor=tk.W, pady=(0, 5))
        
        self.lbl_dimensions = ttk.Label(info_frame, text="Dimensions: N/A")
        self.lbl_dimensions.pack(anchor=tk.W, pady=(0, 5))
        
        self.lbl_size = ttk.Label(info_frame, text="Size: N/A")
        self.lbl_size.pack(anchor=tk.W, pady=(0, 5))
        
        self.lbl_compression = ttk.Label(info_frame, text="Compression: N/A")
        self.lbl_compression.pack(anchor=tk.W, pady=(0, 5))
        
        self.lbl_psnr = ttk.Label(info_frame, text="PSNR: N/A")
        self.lbl_psnr.pack(anchor=tk.W, pady=(0, 5))
        
        # Separator
        ttk.Separator(info_frame).pack(fill=tk.X, pady=5)
        
        # System info
        ttk.Label(info_frame, text="System Information", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        self.lbl_cpu = ttk.Label(info_frame, text="CPU: N/A")
        self.lbl_cpu.pack(anchor=tk.W)
        
        self.lbl_memory = ttk.Label(info_frame, text="Memory: N/A")
        self.lbl_memory.pack(anchor=tk.W)
        
    def create_histogram_panel(self, parent):
        """Create the histogram visualization panel"""
        hist_frame = ttk.LabelFrame(parent, text="Histogram", padding=10)
        hist_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(3, 2), dpi=80)
        self.hist_fig.patch.set_facecolor(self.style['bg'])
        self.hist_ax.set_facecolor(self.style['bg'])
        
        # Embed in Tkinter
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Default empty histogram
        self.hist_ax.text(0.5, 0.5, 'No image loaded', 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.hist_ax.transAxes)
        self.hist_ax.set_xticks([])
        self.hist_ax.set_yticks([])
        self.hist_canvas.draw()
        
    def create_image_display(self, parent):
        """Create the image display area with tabs"""
        # Notebook for tabs
        self.display_notebook = ttk.Notebook(parent)
        self.display_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Original image tab
        self.original_tab = ttk.Frame(self.display_notebook)
        self.display_notebook.add(self.original_tab, text="Original Image")
        
        # Processed image tab
        self.processed_tab = ttk.Frame(self.display_notebook)
        self.display_notebook.add(self.processed_tab, text="Processed Image")
        
        # Comparison tab (will be added when needed)
        
        # Create canvas for original image
        self.original_canvas = tk.Canvas(self.original_tab, bg='white', bd=0, highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for processed image
        self.processed_canvas = tk.Canvas(self.processed_tab, bg='white', bd=0, highlightthickness=0)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.original_scroll_x = ttk.Scrollbar(self.original_tab, orient=tk.HORIZONTAL, command=self.original_canvas.xview)
        self.original_scroll_y = ttk.Scrollbar(self.original_tab, orient=tk.VERTICAL, command=self.original_canvas.yview)
        self.original_canvas.configure(xscrollcommand=self.original_scroll_x.set, yscrollcommand=self.original_scroll_y.set)
        
        self.processed_scroll_x = ttk.Scrollbar(self.processed_tab, orient=tk.HORIZONTAL, command=self.processed_canvas.xview)
        self.processed_scroll_y = ttk.Scrollbar(self.processed_tab, orient=tk.VERTICAL, command=self.processed_canvas.yview)
        self.processed_canvas.configure(xscrollcommand=self.processed_scroll_x.set, yscrollcommand=self.processed_scroll_y.set)
        
        # Only show scrollbars when needed
        self.original_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.original_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.processed_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.processed_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Default empty image display
        self.show_default_display()
        
    def create_status_bar(self):
        """Create the status bar at the bottom of the window"""
        self.status_bar = ttk.Frame(self.root, height=22)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status message
        self.status_message = tk.StringVar(value="Ready")
        ttk.Label(self.status_bar, textvariable=self.status_message, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_bar, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # Hide progress bar initially
        self.progress_bar.pack_forget()
        
    def show_default_display(self):
        """Show default content when no image is loaded"""
        self.original_canvas.delete('all')
        self.processed_canvas.delete('all')
        
        # Original canvas
        self.original_canvas.create_text(
            self.original_canvas.winfo_width()/2, 
            self.original_canvas.winfo_height()/2,
            text="Open an image to begin processing\n(Supported formats: JPG, PNG, WEBP, BMP)",
            font=('Segoe UI', 12),
            fill='gray',
            justify=tk.CENTER
        )
        
        # Processed canvas
        self.processed_canvas.create_text(
            self.processed_canvas.winfo_width()/2, 
            self.processed_canvas.winfo_height()/2,
            text="Processed image will appear here\nafter applying DCT compression",
            font=('Segoe UI', 12),
            fill='gray',
            justify=tk.CENTER
        )
        
    def update_compression_label(self):
        """Update the compression level label when slider changes"""
        level = self.compression_level.get()
        self.lbl_compression_value.config(text=f"{level}%")
        
    def open_image(self):
        """Open an image file and display it"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("WebP files", "*.webp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select an image",
            filetypes=filetypes
        )
        
        if filename:
            self.load_image(filename)
            
    def load_image(self, filename):
        """Load an image from file and update the UI"""
        try:
            # Read image with OpenCV
            self.original_image = cv2.imread(filename)
            
            if self.original_image is None:
                raise ValueError("Unable to read image file")
                
            # Convert to RGB for display
            img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Get file info
            self.filename = filename
            self.file_size_before = os.path.getsize(filename) / 1024  # KB
            dimensions = self.original_image.shape
            height, width = dimensions[0], dimensions[1]
            
            # Update UI
            self.display_image(self.original_canvas, img_rgb)
            self.update_image_info(filename, width, height, self.file_size_before)
            self.update_histogram(self.original_image)
            
            # Enable relevant buttons
            self.toolbar_buttons['dct'].state(['!disabled'])
            self.toolbar_buttons['save'].state(['disabled'])
            self.toolbar_buttons['compare'].state(['disabled'])
            
            # Add to recent files
            self.add_to_recent_files(filename)
            
            # Update status
            self.status_message.set(f"Loaded: {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_message.set("Error loading image")
            
    def display_image(self, canvas, image):
        """Display an image on the specified canvas"""
        canvas.delete('all')
        
        # Convert to PIL Image
        pil_img = Image.fromarray(image)
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Calculate aspect ratio
        img_width, img_height = pil_img.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize image
        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_img)
        
        # Display centered
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
        
        # Configure scroll region
        canvas.configure(scrollregion=(0, 0, new_width, new_height))
        
    def update_image_info(self, filename, width, height, size_kb):
        """Update the image information panel"""
        # Basic info
        self.lbl_filename.config(text=f"File: {os.path.basename(filename)}")
        self.lbl_dimensions.config(text=f"Dimensions: {width} x {height} px")
        self.lbl_size.config(text=f"Size: {size_kb:.2f} KB")
        
        # Clear processed info
        self.lbl_compression.config(text="Compression: N/A")
        self.lbl_psnr.config(text="PSNR: N/A")
        
    def update_histogram(self, image):
        """Update the histogram display"""
        # Clear previous histogram
        self.hist_ax.clear()
        
        # Convert to grayscale if color image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Plot histogram
        self.hist_ax.plot(hist, color='#3a7ca5')
        self.hist_ax.set_title('Intensity Histogram', fontsize=10)
        self.hist_ax.set_xlabel('Pixel Value', fontsize=8)
        self.hist_ax.set_ylabel('Frequency', fontsize=8)
        self.hist_ax.grid(True, alpha=0.3)
        
        # Adjust layout
        self.hist_fig.tight_layout()
        self.hist_canvas.draw()
        
    def apply_dct(self):
        """Apply DCT compression to the image"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded to process")
            return
            
        # Get processing parameters
        compression_level = self.compression_level.get() / 100.0
        block_size = int(self.block_size.get().split('x')[0])
        color_space = self.color_space.get()
        quantization = self.quantization.get()
        
        # Show progress
        self.show_progress(True)
        self.status_message.set("Applying DCT compression...")
        
        # Process in a separate thread
        threading.Thread(
            target=self._apply_dct_thread,
            args=(compression_level, block_size, color_space, quantization),
            daemon=True
        ).start()
        
    def _apply_dct_thread(self, compression_level, block_size, color_space, quantization):
        """Thread function for DCT processing"""
        try:
            # Start timer
            start_time = time.time()
            
            # Process image
            self.progress_var.set(10)
            
            # Convert color space if needed
            if color_space == "YCbCr":
                img_ycbcr = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2YCrCb)
                channels = cv2.split(img_ycbcr)
            elif color_space == "Grayscale":
                img_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                channels = [img_gray]
            else:  # RGB
                channels = cv2.split(self.original_image)
                
            self.progress_var.set(20)
            
            # Process each channel
            processed_channels = []
            dct_coeffs = []
            
            for i, channel in enumerate(channels):
                # Apply DCT block processing
                processed, coeffs = self.process_channel_dct(
                    channel, 
                    block_size, 
                    compression_level,
                    quantization
                )
                processed_channels.append(processed)
                dct_coeffs.append(coeffs)
                self.progress_var.set(20 + (i+1)*60/len(channels))
                
            # Merge channels
            if len(processed_channels) == 1:
                processed_img = processed_channels[0]
            else:
                processed_img = cv2.merge(processed_channels)
                
            # Convert back to original color space
            if color_space == "YCbCr":
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_YCrCb2BGR)
            elif color_space == "Grayscale":
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                
            self.processed_image = processed_img
            self.dct_coefficients = dct_coeffs
            
            # Calculate metrics
            self.calculate_metrics()
            
            # Update UI in main thread
            self.root.after(0, self._update_after_dct)
            
            # Log processing time
            elapsed = time.time() - start_time
            self.status_message.set(f"Processing completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))
            self.status_message.set("Processing failed")
            
        finally:
            self.root.after(0, lambda: self.show_progress(False))
            
    def process_channel_dct(self, channel, block_size, compression_level, quantization):
        """Process a single channel with DCT"""
        height, width = channel.shape
        processed = np.zeros_like(channel, dtype=np.float32)
        coeffs = []
        
        # Create quantization matrix
        if quantization == "Standard JPEG":
            q_matrix = self.create_jpeg_quantization_matrix(block_size)
        elif quantization == "Uniform":
            q_matrix = np.ones((block_size, block_size), dtype=np.float32)
        else:  # Custom
            q_matrix = self.create_custom_quantization_matrix(block_size)
            
        # Scale quantization matrix based on compression level
        q_matrix = q_matrix * (1.0 + (1.0 - compression_level) * 10)
        
        # Process in blocks
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Get block
                block = channel[y:y+block_size, x:x+block_size].astype(np.float32)
                
                # Subtract 128 for DCT
                block -= 128
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Quantize coefficients
                quantized = np.round(dct_block / q_matrix[:block.shape[0], :block.shape[1]])
                
                # Store coefficients for visualization
                coeffs.append(quantized.copy())
                
                # Apply inverse quantization
                dequantized = quantized * q_matrix[:block.shape[0], :block.shape[1]]
                
                # Apply IDCT
                idct_block = cv2.idct(dequantized)
                
                # Add 128 back
                idct_block += 128
                
                # Clip values to valid range
                idct_block = np.clip(idct_block, 0, 255)
                
                # Put back into image
                processed[y:y+block_size, x:x+block_size] = idct_block
                
        return processed.astype(np.uint8), coeffs
        
    def create_jpeg_quantization_matrix(self, size):
        """Create a JPEG-like quantization matrix"""
        if size == 8:
            # Standard JPEG luminance quantization table
            q = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ], dtype=np.float32)
        else:
            # Create a scaled version for other block sizes
            base_q = np.array([[16, 11, 10, 16], 
                              [12, 12, 14, 19], 
                              [14, 13, 16, 24], 
                              [14, 17, 22, 29]], dtype=np.float32)
            q = cv2.resize(base_q, (size, size), interpolation=cv2.INTER_LINEAR)
            q = np.clip(q, 1, 255)
            
        return q
        
    def create_custom_quantization_matrix(self, size):
        """Create a custom quantization matrix that preserves more low frequencies"""
        # Create a matrix that increases quantization step with frequency
        matrix = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                # Distance from DC coefficient (top-left corner)
                distance = np.sqrt(i**2 + j**2)
                # Normalize to 0-1 range
                normalized = distance / np.sqrt(2*(size**2))
                # Create quantization value
                matrix[i,j] = 1 + normalized * 50  # Range from 1 to 51
                
        return matrix
        
    def calculate_metrics(self):
        """Calculate compression metrics"""
        # Calculate PSNR
        if len(self.original_image.shape) == 3 and len(self.processed_image.shape) == 3:
            # Color image
            mse = np.mean((self.original_image - self.processed_image) ** 2)
            if mse == 0:
                self.psnr_value = float('inf')
            else:
                max_pixel = 255.0
                self.psnr_value = 20 * log10(max_pixel / sqrt(mse))
        else:
            # Grayscale image
            original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            processed_gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            mse = np.mean((original_gray - processed_gray) ** 2)
            if mse == 0:
                self.psnr_value = float('inf')
            else:
                max_pixel = 255.0
                self.psnr_value = 20 * log10(max_pixel / sqrt(mse))
                
        # Calculate compression ratio (simulated)
        self.compression_ratio = 1.0 / (1.0 + (100 - self.compression_level.get()) / 100.0)
        
    def _update_after_dct(self):
        """Update UI after DCT processing completes"""
        # Display processed image
        processed_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        self.display_image(self.processed_canvas, processed_rgb)
        
        # Update histogram
        self.update_histogram(self.processed_image)
        
        # Update info panel
        self.lbl_compression.config(text=f"Compression: {self.compression_level.get()}%")
        self.lbl_psnr.config(text=f"PSNR: {self.psnr_value:.2f} dB")
        
        # Enable save and compare buttons
        self.toolbar_buttons['save'].state(['!disabled'])
        self.toolbar_buttons['compare'].state(['!disabled'])
        
        # Add to history
        self.add_to_history()
        
    def save_image(self):
        """Save the processed image to file"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
            
        filetypes = [
            ("JPEG", "*.jpg"),
            ("PNG", "*.png"),
            ("WebP", "*.webp"),
            ("BMP", "*.bmp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=filetypes,
            title="Save processed image"
        )
        
        if filename:
            try:
                # Determine format from extension
                ext = os.path.splitext(filename)[1].lower()
                
                # Save with appropriate parameters
                if ext in ('.jpg', '.jpeg'):
                    cv2.imwrite(filename, self.processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                elif ext == '.png':
                    cv2.imwrite(filename, self.processed_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
                elif ext == '.webp':
                    cv2.imwrite(filename, self.processed_image, [int(cv2.IMWRITE_WEBP_QUALITY), 90])
                else:
                    cv2.imwrite(filename, self.processed_image)
                    
                # Update file size info
                self.file_size_after = os.path.getsize(filename) / 1024  # KB
                compression_ratio = self.file_size_before / self.file_size_after
                
                # Show success message
                messagebox.showinfo(
                    "Success", 
                    f"Image saved successfully\n"
                    f"Original size: {self.file_size_before:.2f} KB\n"
                    f"Compressed size: {self.file_size_after:.2f} KB\n"
                    f"Compression ratio: {compression_ratio:.2f}:1"
                )
                
                self.status_message.set(f"Saved: {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
                self.status_message.set("Error saving image")
                
    def show_comparison(self):
        """Show a side-by-side comparison of original and processed images"""
        if self.original_image is None or self.processed_image is None:
            messagebox.showwarning("Warning", "Both original and processed images are required for comparison")
            return
            
        # Create comparison tab if it doesn't exist
        if not hasattr(self, 'comparison_tab'):
            self.comparison_tab = ttk.Frame(self.display_notebook)
            self.display_notebook.add(self.comparison_tab, text="Comparison")
            
            # Create canvas for comparison
            self.comparison_canvas = tk.Canvas(self.comparison_tab, bg='white', bd=0, highlightthickness=0)
            self.comparison_canvas.pack(fill=tk.BOTH, expand=True)
            
            # Scrollbars
            self.comparison_scroll_x = ttk.Scrollbar(self.comparison_tab, orient=tk.HORIZONTAL, command=self.comparison_canvas.xview)
            self.comparison_scroll_y = ttk.Scrollbar(self.comparison_tab, orient=tk.VERTICAL, command=self.comparison_canvas.yview)
            self.comparison_canvas.configure(xscrollcommand=self.comparison_scroll_x.set, yscrollcommand=self.comparison_scroll_y.set)
            
            self.comparison_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            self.comparison_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            
        # Switch to comparison tab
        self.display_notebook.select(self.comparison_tab)
        
        # Create comparison image
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Images
        original_pil = Image.fromarray(original_rgb)
        processed_pil = Image.fromarray(processed_rgb)
        
        # Resize to same dimensions (take min dimensions)
        width = min(original_pil.width, processed_pil.width)
        height = min(original_pil.height, processed_pil.height)
        original_pil = original_pil.resize((width, height), Image.LANCZOS)
        processed_pil = processed_pil.resize((width, height), Image.LANCZOS)
        
        # Create comparison image
        comparison = Image.new('RGB', (width * 2 + 10, height))
        comparison.paste(original_pil, (0, 0))
        comparison.paste(processed_pil, (width + 10, 0))
        
        # Add divider and labels
        draw = ImageDraw.Draw(comparison)
        draw.line([(width + 5, 0), (width + 5, height)], fill='gray', width=2)
        
        # Convert to PhotoImage
        self.comparison_photo = ImageTk.PhotoImage(comparison)
        
        # Display on canvas
        self.comparison_canvas.delete('all')
        self.comparison_canvas.create_image(0, 0, anchor=tk.NW, image=self.comparison_photo)
        
        # Add labels
        self.comparison_canvas.create_text(
            width//2, 20,
            text="Original",
            font=('Segoe UI', 12, 'bold'),
            fill='white',
            anchor=tk.N
        )
        
        self.comparison_canvas.create_text(
            width + 10 + width//2, 20,
            text="Processed",
            font=('Segoe UI', 12, 'bold'),
            fill='white',
            anchor=tk.N
        )
        
        # Configure scroll region
        self.comparison_canvas.configure(scrollregion=(0, 0, width*2 + 10, height))
        
    def show_dct_coefficients(self):
        """Visualize the DCT coefficients"""
        if self.dct_coefficients is None:
            messagebox.showwarning("Warning", "No DCT coefficients available. Process an image first.")
            return
            
        # Create a new window
        coeff_window = tk.Toplevel(self.root)
        coeff_window.title("DCT Coefficients Visualization")
        coeff_window.geometry("800x600")
        
        # Create notebook for each channel
        notebook = ttk.Notebook(coeff_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # For each channel, create a tab with coefficient visualization
        for i, channel_coeffs in enumerate(self.dct_coefficients):
            # Create frame for this channel
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=f"Channel {i+1}" if len(self.dct_coefficients) > 1 else "Coefficients")
            
            # Create canvas for coefficient visualization
            canvas = tk.Canvas(frame, bg='white')
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Inner frame for content
            inner_frame = ttk.Frame(canvas)
            canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)
            
            # Get first few blocks for visualization
            num_blocks = min(5, len(channel_coeffs))
            
            for j in range(num_blocks):
                # Create figure for this block
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Original coefficients
                ax1.imshow(np.abs(channel_coeffs[j]), cmap='hot', interpolation='nearest')
                ax1.set_title(f"Block {j+1} - Original Coefficients")
                
                # Thresholded coefficients (only significant ones)
                threshold = np.percentile(np.abs(channel_coeffs[j]), 75)
                thresholded = np.where(np.abs(channel_coeffs[j]) > threshold, channel_coeffs[j], 0)
                ax2.imshow(np.abs(thresholded), cmap='hot', interpolation='nearest')
                ax2.set_title(f"Block {j+1} - Significant Coefficients")
                
                fig.tight_layout()
                
                # Embed in Tkinter
                canvas_fig = FigureCanvasTkAgg(fig, master=inner_frame)
                canvas_fig.draw()
                canvas_fig.get_tk_widget().pack(fill=tk.X, padx=5, pady=5)
                
            # Update scroll region
            inner_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox('all'))
            
    def show_frequency_domain(self):
        """Show the frequency domain representation"""
        if self.dct_coefficients is None:
            messagebox.showwarning("Warning", "No DCT coefficients available. Process an image first.")
            return
            
        # Create a new window
        freq_window = tk.Toplevel(self.root)
        freq_window.title("Frequency Domain Visualization")
        freq_window.geometry("800x600")
        
        # Create notebook for each channel
        notebook = ttk.Notebook(freq_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # For each channel, create a tab with frequency visualization
        for i, channel_coeffs in enumerate(self.dct_coefficients):
            # Create frame for this channel
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=f"Channel {i+1}" if len(self.dct_coefficients) > 1 else "Frequency Domain")
            
            # Calculate average magnitude of coefficients
            avg_magnitude = np.mean([np.abs(block) for block in channel_coeffs], axis=0)
            
            # Create figure
            fig = plt.Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Display as heatmap
            cax = ax.imshow(np.log1p(avg_magnitude), cmap='viridis', interpolation='nearest')
            fig.colorbar(cax, ax=ax, label='Log Magnitude')
            
            ax.set_title("Average DCT Coefficient Magnitude (Log Scale)")
            ax.set_xlabel("Horizontal Frequency")
            ax.set_ylabel("Vertical Frequency")
            
            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
    def show_progress(self, show):
        """Show or hide the progress bar"""
        if show:
            self.progress_bar.pack(side=tk.RIGHT, padx=5)
            self.progress_var.set(0)
        else:
            self.progress_bar.pack_forget()
            
    def add_to_history(self):
        """Add current processing to history"""
        if self.original_image is not None and self.processed_image is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.history.append({
                'timestamp': timestamp,
                'compression': self.compression_level.get(),
                'block_size': self.block_size.get(),
                'color_space': self.color_space.get(),
                'psnr': self.psnr_value,
                'original': self.original_image.copy(),
                'processed': self.processed_image.copy()
            })
            
            # Enable undo button
            self.toolbar_buttons['undo'].state(['!disabled'])
            
    def undo_action(self):
        """Undo the last processing step"""
        if len(self.history) > 0:
            # Get last state
            last_state = self.history.pop()
            
            # Restore processed image
            self.processed_image = last_state['processed']
            processed_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.processed_canvas, processed_rgb)
            
            # Update info
            self.lbl_compression.config(text=f"Compression: {last_state['compression']}%")
            self.lbl_psnr.config(text=f"PSNR: {last_state['psnr']:.2f} dB")
            
            # Update histogram
            self.update_histogram(self.processed_image)
            
            # Enable redo button
            self.toolbar_buttons['redo'].state(['!disabled'])
            
            # Disable undo if no more history
            if len(self.history) == 0:
                self.toolbar_buttons['undo'].state(['disabled'])
                
    def redo_action(self):
        """Redo the last undone action"""
        # Not implemented in this version
        pass
        
    def add_to_recent_files(self, filename):
        """Add a file to the recent files list"""
        if 'recent_files' not in self.settings:
            self.settings['recent_files'] = []
            
        # Remove if already exists
        if filename in self.settings['recent_files']:
            self.settings['recent_files'].remove(filename)
            
        # Add to beginning
        self.settings['recent_files'].insert(0, filename)
        
        # Keep only last 10 files
        if len(self.settings['recent_files']) > 10:
            self.settings['recent_files'] = self.settings['recent_files'][:10]
            
        # Save settings
        self.save_settings()
        
    def show_recent_files(self):
        """Show a menu of recent files"""
        if 'recent_files' not in self.settings or not self.settings['recent_files']:
            messagebox.showinfo("Recent Files", "No recent files available")
            return
            
        # Create menu
        menu = tk.Menu(self.root, tearoff=0)
        
        for i, filename in enumerate(self.settings['recent_files']):
            menu.add_command(
                label=f"{i+1}. {os.path.basename(filename)}",
                command=lambda f=filename: self.load_image(f)
            )
            
        # Add clear option
        menu.add_separator()
        menu.add_command(label="Clear Recent Files", command=self.clear_recent_files)
        
        # Show menu
        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()
            
    def clear_recent_files(self):
        """Clear the recent files list"""
        if 'recent_files' in self.settings:
            self.settings['recent_files'] = []
            self.save_settings()
            messagebox.showinfo("Recent Files", "Recent files list cleared")
            
    def show_preferences(self):
        """Show the preferences dialog"""
        prefs_window = tk.Toplevel(self.root)
        prefs_window.title("Preferences")
        prefs_window.geometry("500x400")
        prefs_window.resizable(False, False)
        
        # Create notebook for different preference sections
        notebook = ttk.Notebook(prefs_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # General tab
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="General")
        
        # Appearance section
        ttk.Label(general_frame, text="Appearance", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        self.theme_var = tk.StringVar(value=self.settings.get('theme', 'light'))
        ttk.Label(general_frame, text="Theme:").pack(anchor=tk.W)
        ttk.Combobox(
            general_frame, 
            textvariable=self.theme_var,
            values=['light', 'dark'],
            state='readonly'
        ).pack(fill=tk.X, pady=(0, 10))
        
        # Performance section
        ttk.Label(general_frame, text="Performance", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        self.threading_var = tk.BooleanVar(value=self.settings.get('use_threading', True))
        ttk.Checkbutton(
            general_frame,
            text="Use multi-threading for processing",
            variable=self.threading_var
        ).pack(anchor=tk.W)
        
        # Processing tab
        processing_frame = ttk.Frame(notebook)
        notebook.add(processing_frame, text="Processing")
        
        # Default settings section
        ttk.Label(processing_frame, text="Default Processing Settings", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        ttk.Label(processing_frame, text="Default Compression Level:").pack(anchor=tk.W)
        self.default_compression = tk.IntVar(value=self.settings.get('default_compression', 50))
        ttk.Scale(
            processing_frame,
            from_=1,
            to=100,
            variable=self.default_compression,
            orient=tk.HORIZONTAL
        ).pack(fill=tk.X)
        
        ttk.Label(processing_frame, text="Default Block Size:").pack(anchor=tk.W, pady=(5, 0))
        self.default_block_size = tk.StringVar(value=self.settings.get('default_block_size', '8x8'))
        ttk.Combobox(
            processing_frame,
            textvariable=self.default_block_size,
            values=['4x4', '8x8', '16x16', '32x32'],
            state='readonly'
        ).pack(fill=tk.X)
        
        # Buttons
        btn_frame = ttk.Frame(prefs_window)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            btn_frame,
            text="Save",
            command=lambda: self.save_preferences(prefs_window),
            style='Success.TButton'
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Cancel",
            command=prefs_window.destroy
        ).pack(side=tk.RIGHT)
        
    def save_preferences(self, window):
        """Save preferences to settings"""
        self.settings['theme'] = self.theme_var.get()
        self.settings['use_threading'] = self.threading_var.get()
        self.settings['default_compression'] = self.default_compression.get()
        self.settings['default_block_size'] = self.default_block_size.get()
        
        self.save_settings()
        window.destroy()
        messagebox.showinfo("Preferences", "Preferences saved. Some changes may require restart to take effect.")
        
    def load_default_settings(self):
        """Load default settings from preferences"""
        if 'default_compression' in self.settings:
            self.compression_level.set(self.settings['default_compression'])
            self.update_compression_label()
            
        if 'default_block_size' in self.settings:
            self.block_size.set(self.settings['default_block_size'])
            
    def load_settings(self):
        """Load settings from file"""
        settings_file = os.path.join(os.path.expanduser('~'), '.dct_compressor_settings.json')
        
        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
            
        return {}
        
    def save_settings(self):
        """Save settings to file"""
        settings_file = os.path.join(os.path.expanduser('~'), '.dct_compressor_settings.json')
        
        try:
            with open(settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass
            
    def start_system_monitor(self):
        """Start periodic system monitoring"""
        self.update_system_info()
        self.root.after(1000, self.start_system_monitor)
        
    def update_system_info(self):
        """Update system information display"""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.lbl_cpu.config(text=f"CPU: {cpu_percent}%")
        
        # Memory usage
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_used = mem.used / (1024 ** 3)  # GB
        mem_total = mem.total / (1024 ** 3)  # GB
        self.lbl_memory.config(text=f"Memory: {mem_percent}% ({mem_used:.1f}/{mem_total:.1f} GB)")
        
    def show_documentation(self):
        """Show documentation in browser"""
        docs_url = "https://en.wikipedia.org/wiki/Discrete_cosine_transform"
        webbrowser.open(docs_url)
        
    def show_about(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self.root)
        about_window.title("About Quantum DCT Compressor")
        about_window.geometry("400x300")
        about_window.resizable(False, False)
        
        # Logo
        logo_frame = ttk.Frame(about_window)
        logo_frame.pack(pady=10)
        
        try:
            logo_img = Image.open(self.resource_path('icon.ico')) if os.path.exists(self.resource_path('icon.ico')) else None
            if logo_img:
                logo_img = logo_img.resize((64, 64), Image.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = ttk.Label(logo_frame, image=logo_photo)
                logo_label.image = logo_photo
                logo_label.pack()
        except Exception:
            pass
            
        # Info
        info_frame = ttk.Frame(about_window)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        ttk.Label(
            info_frame,
            text="Quantum DCT Image Compressor",
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=(0, 10))
        
        ttk.Label(
            info_frame,
            text="Version 1.0\n\n"
                 "A powerful tool for image compression using\n"
                 "Discrete Cosine Transform (DCT) algorithm.\n\n"
                 "© 2023 Quantum Imaging Technologies",
            justify=tk.CENTER
        ).pack()
        
        # Close button
        btn_frame = ttk.Frame(about_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Close",
            command=about_window.destroy,
            style='Accent.TButton'
        ).pack(side=tk.RIGHT)
        
    def on_exit(self):
        """Handle application exit"""
        self.save_settings()
        self.root.destroy()
        
if __name__ == '__main__':
    root = tk.Tk()
    
    # Set Windows 10/11 theme if available
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    app = DCTCompressorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()