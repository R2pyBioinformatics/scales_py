"""Python port of the R scales package — scale functions for visualization."""

__version__ = "1.4.0.9000"
__r_commit__ = "04fc333"

# -- Utilities ---------------------------------------------------------------
from ._utils import *  # noqa: F401,F403

# -- Bounds / out-of-bounds --------------------------------------------------
from .bounds import *  # noqa: F401,F403

# -- Transforms --------------------------------------------------------------
from .transforms import *  # noqa: F401,F403

# -- Breaks ------------------------------------------------------------------
from .breaks import *  # noqa: F401,F403
from .breaks_log import *  # noqa: F401,F403
from .minor_breaks import *  # noqa: F401,F403

# -- Palettes ----------------------------------------------------------------
from .palettes import *  # noqa: F401,F403

# -- Colour utilities --------------------------------------------------------
from .colour_ramp import *  # noqa: F401,F403
from .colour_manip import *  # noqa: F401,F403
from .colour_mapping import *  # noqa: F401,F403

# -- Labels ------------------------------------------------------------------
from .labels import *  # noqa: F401,F403

# -- Range -------------------------------------------------------------------
from .range import *  # noqa: F401,F403

# -- Scale helpers -----------------------------------------------------------
from .scale_continuous import *  # noqa: F401,F403
from .scale_discrete import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# Comprehensive public API
# ---------------------------------------------------------------------------
__all__ = [
    # _utils
    "zero_range",
    "expand_range",
    "rescale_common",
    "recycle_common",
    "fullseq",
    "round_any",
    "offset_by",
    "precision",
    "demo_continuous",
    "demo_log10",
    "demo_discrete",
    "demo_datetime",
    "demo_time",
    "demo_timespan",
    # bounds
    "rescale",
    "rescale_mid",
    "rescale_max",
    "rescale_none",
    "censor",
    "squish",
    "squish_infinite",
    "discard",
    "oob_censor",
    "oob_censor_any",
    "oob_squish",
    "oob_squish_any",
    "oob_squish_infinite",
    "oob_keep",
    "oob_discard",
    "trim_to_domain",
    "trans_range",      # R alias: trans_range <- trim_to_domain
    # transforms – core API
    "Transform",
    "new_transform",
    "is_transform",
    "as_transform",
    "trans_breaks",
    "trans_format",
    # transforms – constructors
    "transform_identity",
    "transform_log",
    "transform_log10",
    "transform_log2",
    "transform_log1p",
    "transform_exp",
    "transform_sqrt",
    "transform_reverse",
    "transform_reciprocal",
    "transform_asinh",
    "transform_asn",
    "transform_atanh",
    "transform_boxcox",
    "transform_modulus",
    "transform_yj",
    "transform_pseudo_log",
    "transform_logit",
    "transform_probit",
    "transform_probability",
    "transform_date",
    "transform_time",
    "transform_timespan",
    "transform_compose",
    # transforms – legacy aliases
    "trans_new",
    "identity_trans",
    "log_trans",
    "log10_trans",
    "log2_trans",
    "log1p_trans",
    "exp_trans",
    "sqrt_trans",
    "reverse_trans",
    "reciprocal_trans",
    "asinh_trans",
    "asn_trans",
    "atanh_trans",
    "boxcox_trans",
    "modulus_trans",
    "yj_trans",
    "pseudo_log_trans",
    "logit_trans",
    "probit_trans",
    "probability_trans",
    "date_trans",
    "time_trans",
    "timespan_trans",
    "transform_hms",
    "hms_trans",
    "compose_trans",
    "is_trans",
    "as_trans",
    # breaks
    "breaks_extended",
    "breaks_pretty",
    "breaks_width",
    "breaks_timespan",
    "breaks_exp",
    "cbreaks",
    "extended_breaks",
    "pretty_breaks",
    # breaks_log
    "breaks_log",
    "log_breaks",       # R alias: log_breaks <- breaks_log
    "minor_breaks_log",
    # minor_breaks
    "minor_breaks_n",
    "minor_breaks_width",
    "regular_minor_breaks",
    # palettes – core classes
    "ContinuousPalette",
    "DiscretePalette",
    # palettes – constructors
    "new_continuous_palette",
    "new_discrete_palette",
    # palettes – testing / getters
    "is_pal",
    "is_continuous_pal",
    "is_discrete_pal",
    "is_colour_pal",
    "is_numeric_pal",
    "palette_nlevels",
    "palette_na_safe",
    "palette_type",
    # palettes – coercion
    "as_discrete_pal",
    "as_continuous_pal",
    "register_palette",
    "get_palette",
    "palette_names",
    "reset_palettes",
    # palettes – discrete factories
    "pal_brewer",
    "pal_hue",
    "pal_viridis",
    "pal_grey",
    "pal_shape",
    "pal_linetype",
    "pal_identity",
    "pal_manual",
    "pal_dichromat",
    # palettes – continuous factories
    "pal_gradient_n",
    "pal_div_gradient",
    "pal_seq_gradient",
    "pal_area",
    "pal_rescale",
    "abs_area",
    # palettes – legacy aliases
    "brewer_pal",
    "hue_pal",
    "viridis_pal",
    "grey_pal",
    "shape_pal",
    "linetype_pal",
    "identity_pal",
    "manual_pal",
    "dichromat_pal",
    "gradient_n_pal",
    "div_gradient_pal",
    "seq_gradient_pal",
    "area_pal",
    "rescale_pal",
    # colour_ramp
    "colour_ramp",
    # colour_manip
    "alpha",
    "muted",
    "col2hcl",
    "show_col",
    "col_mix",
    "col_shift",
    "col_lighter",
    "col_darker",
    "col_saturate",
    # colour_mapping
    "col_numeric",
    "col_bin",
    "col_quantile",
    "col_factor",
    # labels – closure factories
    "label_number",
    "label_comma",
    "label_percent",
    "label_dollar",
    "label_currency",
    "label_scientific",
    "label_bytes",
    "label_ordinal",
    "label_pvalue",
    "label_date",
    "label_date_short",
    "label_time",
    "label_timespan",
    "label_wrap",
    "label_glue",
    "label_parse",
    "label_math",
    "label_log",
    "label_number_auto",
    "label_number_si",
    "label_dictionary",
    "compose_label",
    # labels – ordinal helpers
    "ordinal_english",
    "ordinal_french",
    "ordinal_spanish",
    # labels – direct formatting functions
    "number",
    "comma",
    "dollar",
    "percent",
    "scientific",
    "ordinal",
    "pvalue",
    # labels – core log formatting
    "format_log",
    # labels – scale cut helpers
    "cut_short_scale",
    "cut_long_scale",
    "cut_time_scale",
    "cut_si",
    # labels – date utilities
    "date_breaks",
    "date_format",
    "time_format",
    # labels – legacy aliases
    "comma_format",
    "dollar_format",
    "percent_format",
    "scientific_format",
    "ordinal_format",
    "pvalue_format",
    "number_format",
    "number_bytes_format",
    "number_bytes",
    "parse_format",
    "math_format",
    "wrap_format",
    "unit_format",
    "format_format",
    "number_options",
    # range
    "Range",
    "ContinuousRange",
    "DiscreteRange",
    # scale_continuous
    "cscale",
    "train_continuous",
    # scale_discrete
    "dscale",
    "train_discrete",
]
