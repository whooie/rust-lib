//! Conveniences and custom styling commands for [`matplotlib`].

use std::path::{ Path, PathBuf };

pub use matplotlib::{
    Mpl,
    Opt,
    GSPos,
    PyValue,
    Run,
    Matplotlib,
    MatplotlibOpts,
    AsPy,
    MplError,
    MplResult,
    commands,
};

pub use matplotlib::serde_json;
pub use serde_json::value::Value;

const COLORS_MATLAB: &[&str] = &[
    "#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30",
    "#4dbeee", "#a2142f",
];

const COLORS_PYTHON: &[&str] = &[
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
];

const COLORS_PLOTLY: &[&str] = &[
    "#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
    "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52",
];

const COLORS_WHOOIE: &[&str] = &[
    "#1f77b4", // blue
    "#d95319", // auburn
    "#edb120", // canary
    "#7e2f8e", // purple
    "#46add9", // cyan
    "#ff7f0e", // tangerine
    "#3d786e", // dark seafoam
    "#505050", // gray
    "#a2142f", // burgundy
    "#bf7878", // dark rose
];

/// Color set selector.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColorSet {
    Matlab,
    Python,
    Plotly,
    Whooie,
}

impl ColorSet {
    /// Return the raw hex color strings for the colorset, all with leading "#".
    pub fn colors(self) -> &'static [&'static str] {
        match self {
            Self::Matlab => COLORS_MATLAB,
            Self::Python => COLORS_PYTHON,
            Self::Plotly => COLORS_PLOTLY,
            Self::Whooie => COLORS_WHOOIE,
        }
    }
}

/// A color associated with a normalized threshold, `0.0..=1.0`.
pub type ColorPt = (f64, &'static str);

const CMAP_HOTCOLD: &[ColorPt] = &[
    (0.000, "#101010"),
    (0.100, "#3f119d"),
    (0.350, "#3967d0"),
    (0.500, "#f0f0f0"),
    (0.625, "#f1b931"),
    (1.000, "#dd0000"),
];

const CMAP_FIREICE: &[ColorPt] = &[
    (0.000, "#2165ac"),
    (0.167, "#68a9cf"),
    (0.333, "#d2e6f1"),
    (0.500, "#f8f8f8"),
    (0.667, "#ffdbc8"),
    (0.833, "#f08a62"),
    (1.000, "#b0172b"),
];

const CMAP_POWERADE: &[ColorPt] = &[
    (0.000, "#542689"),
    (0.167, "#9a8dc2"),
    (0.333, "#d9daec"),
    (0.500, "#f8f8f8"),
    (0.667, "#d2e6f1"),
    (0.833, "#68a9cf"),
    (1.000, "#2165ac"),
];

const CMAP_FLORAL: &[ColorPt] = &[
    (0.000, "#35c9a5"),
    (0.167, "#5cbea7"),
    (0.333, "#80b4a8"),
    (0.500, "#a8a8a8"),
    (0.667, "#c2a1a8"),
    (0.833, "#e099a9"),
    (1.000, "#fd8fa8"),
];

const CMAP_BLUEHOT: &[ColorPt] = &[
    (0.000, "#000000"),
    (0.450, "#3b4568"),
    (0.600, "#586186"),
    (0.700, "#939cc4"),
    (1.000, "#f8f8f8"),
];

const CMAP_CYBORG: &[ColorPt] = &[
    (0.000, "#101010"),
    (0.100, "#3967d0"),
    (1.000, "#dd0000"),
];

const CMAP_SPORT: &[ColorPt] = &[
    (0.000, "#0c5ab3"),
    (0.125, "#0099e6"),
    (0.250, "#23acf7"),
    (0.500, "#9b74be"),
    (0.750, "#fd6810"),
    (0.875, "#e62600"),
    (1.000, "#b30003"),
];

const CMAP_VIBRANT: &[ColorPt] = &[
    (0.000, "#101010"),
    (0.050, "#012d5e"),
    (0.125, "#0039a7"),
    (0.250, "#1647cf"),
    (0.375, "#6646ff"),
    (0.500, "#bc27ff"),
    (0.600, "#dc47af"),
    (0.800, "#f57548"),
    (0.900, "#f19e00"),
    (0.950, "#fbb800"),
    (1.000, "#fec800"),
];

const CMAP_ARTSY: &[ColorPt] = &[
    (0.000, "#1f0109"),
    (0.034, "#1f0110"),
    (0.069, "#230211"),
    (0.103, "#250816"),
    (0.138, "#270b1b"),
    (0.172, "#250f1d"),
    (0.207, "#251521"),
    (0.241, "#251a25"),
    (0.276, "#2c1b28"),
    (0.310, "#271d2b"),
    (0.345, "#24202d"),
    (0.379, "#232632"),
    (0.414, "#212d32"),
    (0.448, "#1e343c"),
    (0.483, "#173e44"),
    (0.517, "#17464a"),
    (0.552, "#104a49"),
    (0.586, "#0e5553"),
    (0.621, "#00635f"),
    (0.655, "#007065"),
    (0.690, "#007a6d"),
    (0.724, "#0e8476"),
    (0.759, "#1c8c7d"),
    (0.793, "#219581"),
    (0.828, "#2f9f8a"),
    (0.862, "#49a890"),
    (0.897, "#60b89d"),
    (0.931, "#7ec8a9"),
    (0.966, "#9ad6b4"),
    (1.000, "#bce6bf"),
];

const CMAP_PIX: &[ColorPt] = &[
    (0.000, "#0d2b45"),
    (0.143, "#16334d"),
    (0.286, "#544e68"),
    (0.429, "#8d697a"),
    (0.571, "#d08159"),
    (0.714, "#ffaa5e"),
    (0.857, "#ffd4a3"),
    (1.000, "#ffecd6"),
];

const CMAP_SUNSET: &[ColorPt] = &[
    (0.000, "#0d0887"),
    (0.111, "#46039f"),
    (0.222, "#7201a8"),
    (0.333, "#9c179e"),
    (0.444, "#bd3786"),
    (0.555, "#d8576b"),
    (0.666, "#ed7953"),
    (0.777, "#fb9f3a"),
    (0.888, "#fdca26"),
    (1.000, "#f0f921"),
];

const CMAP_TOPOGRAPHY: &[ColorPt] = &[
    (0.000, "#173363"),
    (0.125, "#1d417f"),
    (0.250, "#3266a7"),
    (0.375, "#4194b8"),
    (0.500, "#63bcc2"),
    (0.625, "#9bd8c2"),
    (0.750, "#d2edc7"),
    (0.875, "#f3fad8"),
    (1.000, "#ffffff"),
];

/// An RGB color triple.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    /// Convert from an RGB hex string.
    ///
    /// Strings can be either three or six hex digits, with or without a leading
    /// "#".
    pub fn from_hex(hex: &str) -> Result<Self, RgbParseError> {
        hex.parse()
    }

    /// Convert to an ordinary RGB hex string.
    ///
    /// Output from this function contains a leading "#".
    pub fn as_string(self) -> String {
        format!("#{:x}{:x}{:x}", self.0, self.1, self.2)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RgbParseError {
    MalformedRgb,
    IntParseError(std::num::ParseIntError),
}

impl From<std::num::ParseIntError> for RgbParseError {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::IntParseError(err)
    }
}

impl std::str::FromStr for Rgb {
    type Err = RgbParseError;

    fn from_str(hex: &str) -> Result<Self, Self::Err> {
        match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16)?;
                let g = u8::from_str_radix(&hex[1..2], 16)?;
                let b = u8::from_str_radix(&hex[2..3], 16)?;
                Ok(Rgb(r * 16, g * 16, b * 16))
            },
            4 if hex.starts_with('#') => {
                let r = u8::from_str_radix(&hex[1..2], 16)?;
                let g = u8::from_str_radix(&hex[2..3], 16)?;
                let b = u8::from_str_radix(&hex[3..4], 16)?;
                Ok(Rgb(r * 16, g * 16, b * 16))
            },
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16)?;
                let g = u8::from_str_radix(&hex[2..4], 16)?;
                let b = u8::from_str_radix(&hex[4..6], 16)?;
                Ok(Rgb(r, g, b))
            },
            7 if hex.starts_with('#') => {
                let r = u8::from_str_radix(&hex[1..3], 16)?;
                let g = u8::from_str_radix(&hex[3..5], 16)?;
                let b = u8::from_str_radix(&hex[5..7], 16)?;
                Ok(Rgb(r, g, b))
            },
            _ => Err(RgbParseError::MalformedRgb),
        }
    }
}

impl std::fmt::Display for Rgb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{:x}{:x}{:x}", self.0, self.1, self.2)
    }
}

/// Color map selector.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColorMap {
    HotCold,
    FireIce,
    Powerade,
    Floral,
    BlueHot,
    Cyborg,
    Sport,
    Vibrant,
    Artsy,
    Pix,
    Sunset,
    Topography,
}

// assume all input is valid
fn color_interp(a: ColorPt, b: ColorPt, x: f64) -> Rgb {
    let (fa, ca) = a;
    let (fb, cb) = b;
    let Rgb(ra, ga, ba) = ca.parse().unwrap();
    let Rgb(rb, gb, bb) = cb.parse().unwrap();
    let d = fb - fa;
    let dx = x - fa;
    let r = (ra as f64 + (rb as f64 - ra as f64) / d * dx).round() as u8;
    let g = (ga as f64 + (gb as f64 - ga as f64) / d * dx).round() as u8;
    let b = (ba as f64 + (bb as f64 - ba as f64) / d * dx).round() as u8;
    Rgb(r, g, b)
}

fn find_bounding(points: &[ColorPt], x: f64) -> Option<(ColorPt, ColorPt)> {
    points.iter().zip(points.iter().skip(1))
        .find(|((fl, _), (fr, _))| (*fl..=*fr).contains(&x))
        .map(|((fl, cl), (fr, cr))| ((*fl, *cl), (*fr, *cr)))
}

impl ColorMap {
    /// Get the name of the color map registered with Matplotlib.
    pub fn name(self) -> &'static str {
        match self {
            Self::HotCold    => "hot-cold",
            Self::FireIce    => "fire-ice",
            Self::Powerade   => "powerade",
            Self::Floral     => "floral",
            Self::BlueHot    => "blue-hot",
            Self::Cyborg     => "cyborg",
            Self::Sport      => "sport",
            Self::Vibrant    => "vibrant",
            Self::Artsy      => "artsy",
            Self::Pix        => "pix",
            Self::Sunset     => "sunset",
            Self::Topography => "topography",
        }
    }

    /// Get the bare thresholds and hex color strings for the color map.
    ///
    /// Each color string contains a leading "#".
    pub fn points(self) -> &'static [ColorPt] {
        match self {
            Self::HotCold    => CMAP_HOTCOLD,
            Self::FireIce    => CMAP_FIREICE,
            Self::Powerade   => CMAP_POWERADE,
            Self::Floral     => CMAP_FLORAL,
            Self::BlueHot    => CMAP_BLUEHOT,
            Self::Cyborg     => CMAP_CYBORG,
            Self::Sport      => CMAP_SPORT,
            Self::Vibrant    => CMAP_VIBRANT,
            Self::Artsy      => CMAP_ARTSY,
            Self::Pix        => CMAP_PIX,
            Self::Sunset     => CMAP_SUNSET,
            Self::Topography => CMAP_TOPOGRAPHY,
        }
    }

    /// Convert a number in the `0.0..=1.0` range to a color. Values outside the
    /// range are clamped to colors associated with the range boundaries.
    pub fn map(self, x: f64) -> Rgb {
        let cpts = self.points();
        if x <= 0.0 {
            cpts.first().unwrap().1.parse().unwrap()
        } else if x >= 1.0 {
            cpts.last().unwrap().1.parse().unwrap()
        } else {
            let (a, b) = find_bounding(cpts, x).unwrap();
            color_interp(a, b, x)
        }
    }

    /// Like [`map`][Self::map], but immediately render the result to a
    /// Matplotlib-compatible string.
    pub fn maps(self, x: f64) -> String {
        self.map(x).as_string()
    }
}

fn as_py_str_list(strs: &[&str]) -> String {
    let mut strs: String =
        strs.iter()
        .fold(
            "[".to_string(),
            |mut acc, s| {
                acc.push('"');
                acc.push_str(s);
                acc.push_str("\", ");
                acc
            },
        );
    strs.push(']');
    strs
}

fn py_linsegcmap(name: &str, cmap: &[ColorPt]) -> String {
    cmap.iter()
        .fold(
            format!("mcolors.LinearSegmentedColormap.from_list(\"{}\", [", name),
            |mut acc, (f, c)| {
                acc += &format!("({}, \"{}\"), ", f, c);
                acc
            },
        )
        + "])"
}

fn py_registercmap(cmap: ColorMap) -> String {
    let name = cmap.name();
    let points = cmap.points();
    format!("matplotlib.colormaps.register({})", py_linsegcmap(name, points))
}

/// All custom styling commands.
///
/// This registers all variants of [`ColorMap`] as `LinearSegmentedColormap`s
/// under the names output by [`ColorMap::name`] and overrides the default color
/// cycle setting in rcParams under `axes.prop_cycle`. The following rcParams
/// keys are also overridden:
///
/// | Key                     | Value            |
/// |:------------------------|:-----------------|
/// | `axes.grid`             | `True`           |
/// | `axes.grid.which`       | `"both"`         |
/// | `axes.linewidth`        | `0.65`           |
/// | `axes.titlesize`        | `"medium"`       |
/// | `errorbar.capsize`      | `1.25`           |
/// | `figure.dpi`            | `500.0`          |
/// | `figure.figsize`        | `[3.375, 2.225]` |
/// | `figure.labelsize`      | `"medium"`       |
/// | `font.size`             | `8.0`            |
/// | `grid.color`            | `"#d8d8d8"`      |
/// | `grid.linewidth`        | `0.5`            |
/// | `image.cmap`            | `"jet"`          |
/// | `image.composite_image` | `False`          |
/// | `legend.borderaxespad`  | `0.25`           |
/// | `legend.borderpad`      | `0.2`            |
/// | `legend.fancybox`       | `False`          |
/// | `legend.fontsize`       | `"x-small"`      |
/// | `legend.framealpha`     | `0.8`            |
/// | `legend.handlelength`   | `1.2`            |
/// | `legend.handletextpad`  | `0.4`            |
/// | `legend.labelspacing`   | `0.25`           |
/// | `lines.linewidth`       | `0.8`            |
/// | `lines.markeredgewidth` | `0.8`            |
/// | `lines.markerfacecolor` | `"white"`        |
/// | `lines.markersize`      | `2.0`            |
/// | `markers.fillstyle`     | `"full"`         |
/// | `savefig.bbox`          | `"tight"`        |
/// | `savefig.pad_inches`    | `0.05`           |
/// | `xtick.direction`       | `"in"`           |
/// | `xtick.major.size`      | `2.0`            |
/// | `xtick.minor.size`      | `1.5`            |
/// | `ytick.direction`       | `"in"`           |
/// | `ytick.major.size`      | `2.0`            |
/// | `ytick.minor.size`      | `1.5`            |
///
/// Requires [`commands::DefPrelude`].
///
/// Prelude: **Yes**
///
/// JSON data: **None**
#[derive(Copy, Clone, Debug)]
pub struct Styling;

impl Matplotlib for Styling {
    fn is_prelude(&self) -> bool { true }

    fn data(&self) -> Option<serde_json::value::Value> { None }

    fn py_cmd(&self) -> String {
        let mut code = String::new();
        code.push_str(&(py_registercmap(ColorMap::HotCold) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::FireIce) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Powerade) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Floral) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::BlueHot) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Cyborg) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Sport) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Vibrant) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Artsy) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Pix) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Sunset) + "\n"));
        code.push_str(&(py_registercmap(ColorMap::Topography) + "\n"));
        code.push_str("from cycler import cycler\n");
        code.push_str(&format!("colors = {}\n", as_py_str_list(COLORS_WHOOIE)));
        code.push_str("plt.rcParams[\"axes.grid\"] = True\n");
        code.push_str("plt.rcParams[\"axes.grid.which\"] = \"both\"\n");
        code.push_str("plt.rcParams[\"axes.linewidth\"] = 0.65\n");
        code.push_str("plt.rcParams[\"axes.prop_cycle\"] = cycler(color=colors)\n");
        code.push_str("plt.rcParams[\"axes.titlesize\"] = \"medium\"\n");
        code.push_str("plt.rcParams[\"errorbar.capsize\"] = 1.25\n");
        code.push_str("plt.rcParams[\"figure.dpi\"] = 500.0\n");
        code.push_str("plt.rcParams[\"figure.figsize\"] = [3.375, 2.225]\n");
        code.push_str("plt.rcParams[\"figure.labelsize\"] = \"medium\"\n");
        code.push_str("plt.rcParams[\"font.size\"] = 8.0\n");
        code.push_str("plt.rcParams[\"grid.color\"] = \"#d8d8d8\"\n");
        code.push_str("plt.rcParams[\"grid.linewidth\"] = 0.5\n");
        code.push_str("plt.rcParams[\"image.cmap\"] = \"jet\"\n");
        code.push_str("plt.rcParams[\"image.composite_image\"] = False\n");
        code.push_str("plt.rcParams[\"legend.borderaxespad\"] = 0.25\n");
        code.push_str("plt.rcParams[\"legend.borderpad\"] = 0.2\n");
        code.push_str("plt.rcParams[\"legend.fancybox\"] = False\n");
        code.push_str("plt.rcParams[\"legend.fontsize\"] = \"x-small\"\n");
        code.push_str("plt.rcParams[\"legend.framealpha\"] = 0.8\n");
        code.push_str("plt.rcParams[\"legend.handlelength\"] = 1.2\n");
        code.push_str("plt.rcParams[\"legend.handletextpad\"] = 0.4\n");
        code.push_str("plt.rcParams[\"legend.labelspacing\"] = 0.25\n");
        code.push_str("plt.rcParams[\"lines.linewidth\"] = 0.8\n");
        code.push_str("plt.rcParams[\"lines.markeredgewidth\"] = 0.8\n");
        code.push_str("plt.rcParams[\"lines.markerfacecolor\"] = \"white\"\n");
        code.push_str("plt.rcParams[\"lines.markersize\"] = 2.0\n");
        code.push_str("plt.rcParams[\"markers.fillstyle\"] = \"full\"\n");
        code.push_str("plt.rcParams[\"savefig.bbox\"] = \"tight\"\n");
        code.push_str("plt.rcParams[\"savefig.pad_inches\"] = 0.05\n");
        code.push_str("plt.rcParams[\"xtick.direction\"] = \"in\"\n");
        code.push_str("plt.rcParams[\"xtick.major.size\"] = 2.0\n");
        code.push_str("plt.rcParams[\"xtick.minor.size\"] = 1.5\n");
        code.push_str("plt.rcParams[\"ytick.direction\"] = \"in\"\n");
        code.push_str("plt.rcParams[\"ytick.major.size\"] = 2.0\n");
        code.push_str("plt.rcParams[\"ytick.minor.size\"] = 1.5\n");
        code
    }
}

/// Set the default font family, provided a path to a suitable font file and a
/// name for the font, via `font.family` in `rcParams`.
///
/// ```text
/// import matplotlib.font_manager as mfont
/// font_entry = mfont.FontEntry(fname={path}, name={name})
/// mfont.fontManager.ttflist.insert(0, font_entry)
/// plt.rcParams["font.family"] = font_entry.name
/// ```
///
/// Prelude: **Yes**
///
/// JSON data: **None**
#[derive(Clone, Debug)]
pub struct SetFont {
    /// Font name.
    pub name: String,
    /// Path to a TTF file.
    pub path: PathBuf,
}

impl SetFont {
    /// Create a new `SetFont`.
    pub fn new<P>(name: &str, path: P) -> Self
    where P: AsRef<Path>
    {
        Self { name: name.to_string(), path: path.as_ref().to_path_buf() }
    }
}

/// Create a new `SetFont`.
pub fn set_font<P>(name: &str, path: P) -> SetFont
where P: AsRef<Path>
{
    SetFont::new(name, path)
}

impl Matplotlib for SetFont {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<serde_json::value::Value> { None }

    fn py_cmd(&self) -> String {
        format!("\
            import matplotlib.font_manager as mfont\n\
            font_entry = mfont.FontEntry(fname={}, name={})\n\
            mfont.fontManager.ttflist.insert(0, font_entry)\n\
            plt.rcParams[\"font.family\"] = font_entry.name",
            self.path.display().to_string().as_py(),
            self.name.as_py(),
        )
    }
}

/// Turn the coordinate grid on or off with a custom style.
///
/// ```text
/// if {onoff}:
///     ax.minorticks_on()
///     ax.grid(True, "major", color="#d8d8d8", zorder=-10)
///     ax.grid(True, "minor", color="#e0e0e0", linestyle=":", zorder=-1)
///     ax.tick_params(which="both", direction="in")
/// else:
///     ax.minorticks_off()
///     ax.grid(False)
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Copy, Clone, Debug)]
pub struct GGrid {
    /// On/off setting.
    pub onoff: bool
}

impl GGrid {
    /// Create a new `GGrid`.
    pub fn new(onoff: bool) -> Self { Self { onoff } }
}

/// Create a new `GGrid`.
pub fn ggrid(onoff: bool) -> GGrid {
    GGrid::new(onoff)
}

impl Matplotlib for GGrid {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<serde_json::value::Value> { None }

    fn py_cmd(&self) -> String {
        format!(
"if {}:
    ax.minorticks_on()
    ax.grid(True, \"major\", color=\"#d8d8d8\", zorder=-10)
    ax.grid(True, \"minor\", color=\"#e0e0e0\", linestyle=\":\", zorder=-1)
    ax.tick_params(which=\"both\", direction=\"in\")
else:
    ax.minorticks_off()
    ax.grid(False)",
            self.onoff.as_py()
        )
    }
}

/// Construct a new [`Mpl`] with [`DefPrelude`][commands::DefPrelude],
/// [`Styling`], and [`DefInit`][commands::DefInit].
pub fn mpl() -> Mpl {
    Mpl::default()
        & commands::DefPrelude
        & Styling
        & commands::DefInit
}

/// Construct a new [`Mpl`] using [`Mpl::new_3d`] with [`Styling`].
pub fn mpl_3d<I>(opts: I) -> Mpl
where I: IntoIterator<Item = Opt>
{
    Mpl::default()
        & commands::DefPrelude
        & Styling
        & commands::Init3D::new().oo(opts)
}

/// Construct a new [`Mpl`] using [`Mpl::new_grid`] with [`Styling`].
pub fn mpl_grid<I>(nrows: usize, ncols: usize, opts: I) -> Mpl
where I: IntoIterator<Item = Opt>
{
    Mpl::default()
        & commands::DefPrelude
        & Styling
        & commands::init_grid(nrows, ncols).oo(opts)
}

/// Construct a new [`Mpl`] using [`Mpl::new_gridspec`] with [`Styling`].
pub fn mpl_gridspec<I, P>(gridspec_kw: I, positions: P) -> Mpl
where
    I: IntoIterator<Item = Opt>,
    P: IntoIterator<Item = GSPos>,
{
    Mpl::default()
        & commands::DefPrelude
        & Styling
        & commands::init_gridspec(gridspec_kw, positions)
}

