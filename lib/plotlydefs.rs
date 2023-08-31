//! Useful definitions and constructs to use `plotly` smoothly. Still under
//! construction!

use plotly::{
    self as pl,
    layout as plla,
    common as plco,
};

pub const DEF_WIDTH: usize = (3.375 * 320.0) as usize;
pub const DEF_HEIGHT: usize = (2.500 * 320.0) as usize;

macro_rules! mkcolors {
    (
        $name:ident : { $(
            $c:ident = $color:literal
        ),+ $(,)? },
        $wayname:ident
    ) => {
        pub enum $name {
            $( $c ),+
        }

        impl plco::color::Color for $name {
            fn to_color(&self) -> plco::color::ColorWrapper {
                return match self { $(
                    &$name::$c
                        => plco::color::ColorWrapper::S($color.to_string())
                ),+ };
            }
        }

        pub fn $wayname() -> Vec<$name> { vec![ $( $name::$c ),+ ] }
    }
}

mkcolors!(
    ColorsMatlab : {
        C0 = "#0072bd",
        C1 = "#d95319",
        C2 = "#edb120",
        C3 = "#7e2f8e",
        C4 = "#77ac30",
        C5 = "#4dbeee",
        C6 = "#a2142f",
    },
    colors_matlab
);

mkcolors!(
    ColorsPython : {
        C0 = "#1f77b4",
        C1 = "#ff7f0e",
        C2 = "#2ca02c",
        C3 = "#d62728",
        C4 = "#9467bd",
        C5 = "#8c564b",
        C6 = "#e377c2",
        C7 = "#7f7f7f",
        C8 = "#bcbd22",
        C9 = "#17becf",
    },
    colors_python
);

mkcolors!(
    ColorsPlotly : {
        C0 = "#636efa",
        C1 = "#ef553b",
        C2 = "#00cc96",
        C3 = "#ab63fa",
        C4 = "#ffa15a",
        C5 = "#19d3f3",
        C6 = "#ff6692",
        C7 = "#b6e880",
        C8 = "#ff97ff",
        C9 = "#fecb52",
    },
    colors_plotly
);

mkcolors!(
    ColorsWhooie : {
        C0 = "#1f77b4", // 0 blue
        C1 = "#d95319", // 1 auburn
        C2 = "#edb120", // 2 canary
        C3 = "#7e2f8e", // 3 purple
        C4 = "#46add9", // 4 cyan
        C5 = "#ff7f0e", // 5 tangerine
        C6 = "#3d786e", // 6 dark seafoam
        C7 = "#505050", // 7 gray
        C8 = "#a2142f", // 8 burgundy
        C9 = "#bf7878", // 9 dark rose
    },
    colors_whooie
);

// macro_rules! mkcolorscale {
//     (
//         $name:ident : { $(
//             ($x:literal, $color:literal)
//         ),+ $(,)? }
//     ) => {
//         pub fn $name() -> plco::ColorScale {
//             return plco::ColorScale::Vector(vec![ $(
//                 plco::ColorScaleElement($x, $color.to_string())
//             ),+ ]);
//         }
//     }
// }
//
// mkcolorscale!(
//     colorscale_hot_cold : {
//         (0.000, "#101010"),
//         (0.100, "#3f119d"),
//         (0.350, "#3967d0"),
//         (0.500, "#f0f0f0"),
//         (0.625, "#f1b931"),
//         (1.000, "#dd0000"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_fire_ice : {
//         (0.000, "#2165ac"),
//         (0.167, "#68a9cf"),
//         (0.333, "#d2e6f1"),
//         (0.500, "#f8f8f8"),
//         (0.667, "#ffdbc8"),
//         (0.833, "#f08a62"),
//         (1.000, "#b0172b"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_powerade : {
//         (0.000, "#542689"),
//         (0.167, "#9a8dc2"),
//         (0.333, "#d9daec"),
//         (0.500, "#f8f8f8"),
//         (0.667, "#d2e6f1"),
//         (0.833, "#68a9cf"),
//         (1.000, "#2165ac"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_floral : {
//         (0.000, "#35c9a5"),
//         (0.167, "#5cbea7"),
//         (0.333, "#80b4a8"),
//         (0.500, "#a8a8a8"),
//         (0.667, "#c2a1a8"),
//         (0.833, "#e099a9"),
//         (1.000, "#fd8fa8"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_plasma : {
//         (0.000, "#000000"),
//         (0.450, "#3b4568"),
//         (0.600, "#586186"),
//         (0.700, "#939cc4"),
//         (1.000, "#f8f8f8"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_cyborg : {
//         (0.000, "#101010"),
//         (0.100, "#3967d0"),
//         (1.000, "#dd0000"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_vibrant : {
//         (0.000, "#101010"),
//         (0.050, "#012d5e"),
//         (0.125, "#0039a7"),
//         (0.250, "#1647cf"),
//         (0.375, "#6646ff"),
//         (0.500, "#bc27ff"),
//         (0.600, "#dc47af"),
//         (0.800, "#f57548"),
//         (0.900, "#f19e00"),
//         (0.950, "#fbb800"),
//         (1.000, "#fec800"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_artsy : {
//         (0.000, "#1f0109"),
//         (0.034, "#1f0110"),
//         (0.069, "#230211"),
//         (0.103, "#250816"),
//         (0.138, "#270b1b"),
//         (0.172, "#250f1d"),
//         (0.207, "#251521"),
//         (0.241, "#251a25"),
//         (0.276, "#2c1b28"),
//         (0.310, "#271d2b"),
//         (0.345, "#24202d"),
//         (0.379, "#232632"),
//         (0.414, "#212d32"),
//         (0.448, "#1e343c"),
//         (0.483, "#173e44"),
//         (0.517, "#17464a"),
//         (0.552, "#104a49"),
//         (0.586, "#0e5553"),
//         (0.621, "#00635f"),
//         (0.655, "#007065"),
//         (0.690, "#007a6d"),
//         (0.724, "#0e8476"),
//         (0.759, "#1c8c7d"),
//         (0.793, "#219581"),
//         (0.828, "#2f9f8a"),
//         (0.862, "#49a890"),
//         (0.897, "#60b89d"),
//         (0.931, "#7ec8a9"),
//         (0.966, "#9ad6b4"),
//         (1.000, "#bce6bf"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_pix : {
//         (0.000, "#0d2b45"),
//         (0.143, "#16334d"),
//         (0.286, "#544e68"),
//         (0.429, "#8d697a"),
//         (0.571, "#d08159"),
//         (0.714, "#ffaa5e"),
//         (0.857, "#ffd4a3"),
//         (1.000, "#ffecd6"),
//     }
// );
//
// mkcolorscale!(
//     colorscale_sunset : {
//         (0.000, "#0d0887"),
//         (0.111, "#46039f"),
//         (0.222, "#7201a8"),
//         (0.333, "#9c179e"),
//         (0.444, "#bd3786"),
//         (0.555, "#d8576b"),
//         (0.666, "#ed7953"),
//         (0.777, "#fb9f3a"),
//         (0.888, "#fdca26"),
//         (1.000, "#f0f921"),
//     }
// );

pub fn def_layout() -> pl::Layout {
    return pl::Layout::new()
        .auto_size(true)
        .colorway(colors_whooie())
        .color_scale(plla::LayoutColorScale::new()
            .sequential(
                plco::ColorScale::Palette(plco::ColorScalePalette::Jet))
            .sequential_minus(
                plco::ColorScale::Palette(plco::ColorScalePalette::Jet))
            .diverging(
                plco::ColorScale::Palette(plco::ColorScalePalette::Bluered))
        )
        .font(plco::Font::new()
            .family("Myriad Pro")
            .size(24)
            .color("black")
        )
        .paper_background_color("white")
        .plot_background_color("#eaeaea")
        .x_axis(plla::Axis::new()
            .visible(true)
            .color("black")
            .show_line(true)
            .line_color("black")
            .line_width(3)
            .show_grid(true)
            .grid_color("white")
            .grid_width(3)
            .zero_line(false)
        )
        .y_axis(plla::Axis::new()
            .visible(true)
            .color("black")
            .show_line(true)
            .line_color("black")
            .line_width(3)
            .show_grid(true)
            .grid_color("white")
            .grid_width(3)
            .zero_line(false)
        )
        ;
}

pub fn def_line() -> plco::Line {
    return plco::Line::new()
        .width(4.0)
        ;
}

pub fn def_marker() -> plco::Marker {
    return plco::Marker::new()
        .symbol(plco::MarkerSymbol::CirleOpen)
        .size(4)
        ;
}

