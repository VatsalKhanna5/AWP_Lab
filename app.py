import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Constants ---
C = 299792458  # Speed of light in m/s

# --- Helper Functions (Conversions) ---
def dbm_to_watts(dbm):
    """Convert power from dBm to Watts."""
    return 10**((dbm - 30) / 10)

def watts_to_dbm(watts):
    """Convert power from Watts to dBm."""
    # Add a small epsilon to avoid log10(0)
    epsilon = 1e-30
    return 10 * np.log10(watts + epsilon) + 30

def dbi_to_linear(dbi):
    """Convert antenna gain from dBi to linear scale."""
    return 10**(dbi / 10)

def linear_to_dbi(linear):
    """Convert antenna gain from linear scale to dBi."""
    epsilon = 1e-30
    return 10 * np.log10(linear + epsilon)

# --- Page 1: Friis Transmission Equation Lab ---
def show_friis_lab():
    """Renders the Streamlit page for the Friis Transmission Equation."""
    st.title("📡 Friis Transmission Equation Lab")
    st.markdown("Explore how power is received from one antenna to another.")

    # Lab notes expander
    with st.expander("🔬 Lab Notes: The Equation", expanded=True):
        st.latex(r'''
        P_r = P_t G_t G_r \left( \frac{\lambda}{4 \pi R} \right)^2
        ''')
        st.markdown(r"""
        Where:
        - $P_r$ = Received Power
        - $P_t$ = Transmitted Power
        - $G_t$ = Transmitter Antenna Gain
        - $G_r$ = Receiver Antenna Gain
        - $\lambda$ = Wavelength ($c/f$)
        - $R$ = Distance between antennas
        """)

    # Create tabs for inputs and plots
    tab_calc, tab_2d, tab_3d = st.tabs(["🎛️ Inputs & Calculation", "📊 2D Plots", "📈 3D Plots"])

    with tab_calc:
        st.subheader("Input Parameters")
        
        # --- Fun Mode: Presets ---
        st.markdown("#### Fun Mode (Presets)")
        preset = st.selectbox("Choose a scenario:", 
                              ("Custom", 
                               "Short-Range WiFi (10m)", 
                               "Bluetooth (5m)", 
                               "Rural Broadband (5km)", 
                               "GEO Satellite Link (35,786km)"))

        # Default values
        defaults = {
            "pt_dbm": 20.0,
            "gt_dbi": 2.0,
            "gr_dbi": 2.0,
            "freq_mhz": 2400.0,
            "dist_km": 0.01
        }

        if preset == "Short-Range WiFi (10m)":
            defaults = {"pt_dbm": 20.0, "gt_dbi": 6.0, "gr_dbi": 2.0, "freq_mhz": 5200.0, "dist_km": 0.01}
        elif preset == "Bluetooth (5m)":
            defaults = {"pt_dbm": 10.0, "gt_dbi": 1.0, "gr_dbi": 1.0, "freq_mhz": 2450.0, "dist_km": 0.005}
        elif preset == "Rural Broadband (5km)":
            defaults = {"pt_dbm": 30.0, "gt_dbi": 17.0, "gr_dbi": 10.0, "freq_mhz": 700.0, "dist_km": 5.0}
        elif preset == "GEO Satellite Link (35,786km)":
            defaults = {"pt_dbm": 40.0, "gt_dbi": 45.0, "gr_dbi": 45.0, "freq_mhz": 12000.0, "dist_km": 35786.0}

        # --- Lab Mode: Manual Inputs ---
        st.markdown("---")
        st.markdown("#### Lab Mode (Manual Inputs)")
        
        col1, col2 = st.columns(2)
        with col1:
            pt_dbm = st.slider("Transmitted Power (Pt) [dBm]", -30.0, 60.0, defaults["pt_dbm"], 0.5)
            gt_dbi = st.slider("Transmitter Gain (Gt) [dBi]", -10.0, 60.0, defaults["gt_dbi"], 0.5)
            gr_dbi = st.slider("Receiver Gain (Gr) [dBi]", -10.0, 60.0, defaults["gr_dbi"], 0.5)
        
        with col2:
            freq_mhz = st.number_input("Frequency (f) [MHz]", 1.0, 100000.0, defaults["freq_mhz"], 100.0, format="%.1f")
            dist_km = st.number_input("Distance (R) [km]", 0.001, 100000.0, defaults["dist_km"], 1.0, format="%.3f")

        # --- Calculation ---
        freq_hz = freq_mhz * 1e6
        lambda_m = C / freq_hz
        dist_m = dist_km * 1000

        pt_w = dbm_to_watts(pt_dbm)
        gt_lin = dbi_to_linear(gt_dbi)
        gr_lin = dbi_to_linear(gr_dbi)

        # Friis Equation in linear scale
        pr_w = pt_w * gt_lin * gr_lin * (lambda_m / (4 * np.pi * dist_m))**2
        
        # Convert back to dBm
        pr_dbm = watts_to_dbm(pr_w)
        
        # Calculate Free Space Path Loss (FSPL)
        # Handle potential log10(0) if dist_m or freq_hz is 0
        if dist_m > 0 and freq_hz > 0:
            fspl_db = 20 * np.log10(dist_m) + 20 * np.log10(freq_hz) + 20 * np.log10((4 * np.pi) / C)
        else:
            fspl_db = np.inf

        st.markdown("---")
        st.subheader("⚡ Results")
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Received Power (Pr)", f"{pr_dbm:.2f} dBm")
        res_col2.metric("Path Loss (FSPL)", f"{fspl_db:.2f} dB")
        
        st.markdown("---")
        st.subheader("📊 Advanced Analysis & Interpretation")
        
        # Link Budget
        eirp_dbm = pt_dbm + gt_dbi
        st.markdown("#### Link Budget Breakdown")
        st.code(f"""
        Transmitted Power (Pt):   {pt_dbm:8.2f} dBm
        + Transmitter Gain (Gt):  {gt_dbi:8.2f} dBi
        ======================================
        EIRP:                     {eirp_dbm:8.2f} dBm
        - Free Space Path Loss:   {fspl_db:8.2f} dB
        + Receiver Gain (Gr):     {gr_dbi:8.2f} dBi
        ======================================
        Received Power (Pr):      {pr_dbm:8.2f} dBm
        """)
        
        # Interpretation
        st.markdown("#### Interpretation")
        if pr_dbm > -80:
            st.success(f"**Excellent Link ({pr_dbm:.2f} dBm):** Signal is very strong. Ideal for high-throughput applications like 4K video streaming or high-speed data backhaul.")
        elif pr_dbm > -100:
            st.info(f"**Reliable Link ({pr_dbm:.2f} dBm):** Good signal strength. Suitable for web browsing, HD streaming, and reliable voice calls.")
        elif pr_dbm > -120:
            st.warning(f"**Weak Link ({pr_dbm:.2f} dBm):** Signal is weak. May only be suitable for low-data-rate applications (e.g., IoT sensors, text messages). Expect dropouts.")
        else:
            st.error(f"**No Link ({pr_dbm:.2f} dBm):** Signal is below a usable threshold. The link is not viable.")


    with tab_2d:
        st.subheader("2D Visualizations")
        
        # Plot 1: Received Power vs. Distance
        st.markdown("#### Received Power vs. Distance")
        dist_range_km = np.linspace(dist_km / 10 if dist_km > 0.1 else 0.001, dist_km * 3, 200)
        dist_range_m = dist_range_km * 1000
        pr_w_range = pt_w * gt_lin * gr_lin * (lambda_m / (4 * np.pi * dist_range_m))**2
        pr_dbm_range = watts_to_dbm(pr_w_range)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dist_range_km, y=pr_dbm_range, mode='lines', name='Received Power'))
        fig1.add_vline(x=dist_km, line_dash="dash", line_color="red", annotation_text="Current Distance")
        fig1.update_layout(title="Received Power (Pr) as Distance Changes",
                           xaxis_title="Distance (km)",
                           yaxis_title="Received Power (dBm)")
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Received Power vs. Frequency
        st.markdown("#### Received Power vs. Frequency")
        freq_range_mhz = np.linspace(freq_mhz / 4 if freq_mhz > 100 else 1, freq_mhz * 3, 200)
        lambda_range = C / (freq_range_mhz * 1e6)
        pr_w_freq_range = pt_w * gt_lin * gr_lin * (lambda_range / (4 * np.pi * dist_m))**2
        pr_dbm_freq_range = watts_to_dbm(pr_w_freq_range)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=freq_range_mhz, y=pr_dbm_freq_range, mode='lines', name='Received Power'))
        fig2.add_vline(x=freq_mhz, line_dash="dash", line_color="red", annotation_text="Current Frequency")
        fig2.update_layout(title="Received Power (Pr) as Frequency Changes",
                           xaxis_title="Frequency (MHz)",
                           yaxis_title="Received Power (dBm)")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Plot 3: Received Power vs. Transmitter Gain
        st.markdown("#### Received Power vs. Transmitter Gain")
        gt_range_dbi = np.linspace(max(-10, gt_dbi - 20), gt_dbi + 20, 200)
        gt_range_lin = dbi_to_linear(gt_range_dbi)
        pr_w_gt_range = pt_w * gt_range_lin * gr_lin * (lambda_m / (4 * np.pi * dist_m))**2
        pr_dbm_gt_range = watts_to_dbm(pr_w_gt_range)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=gt_range_dbi, y=pr_dbm_gt_range, mode='lines', name='Received Power'))
        fig3.add_vline(x=gt_dbi, line_dash="dash", line_color="red", annotation_text="Current Gt")
        fig3.update_layout(title="Received Power (Pr) as Transmitter Gain Changes",
                           xaxis_title="Transmitter Gain (dBi)",
                           yaxis_title="Received Power (dBm)")
        st.plotly_chart(fig3, use_container_width=True)

    with tab_3d:
        st.subheader("3D Visualization")
        st.markdown("#### Variant 1: Received Power vs. Distance and Frequency")

        # Create 3D data
        dist_3d_km = np.linspace(dist_km / 10 if dist_km > 0.1 else 0.001, dist_km * 3, 50)
        freq_3d_mhz = np.linspace(freq_mhz / 4 if freq_mhz > 100 else 1, freq_mhz * 3, 50)
        
        # Create meshgrid
        D_km, F_mhz = np.meshgrid(dist_3d_km, freq_3d_mhz)
        D_m = D_km * 1000
        F_hz = F_mhz * 1e6
        
        # Calculate Wavelength and Pr in 3D
        LAMBDA_m = C / F_hz
        PR_w_3D = pt_w * gt_lin * gr_lin * (LAMBDA_m / (4 * np.pi * D_m))**2
        PR_dbm_3D = watts_to_dbm(PR_w_3D)

        fig3d = go.Figure(data=[go.Surface(
            z=PR_dbm_3D, 
            x=D_km, 
            y=F_mhz,
            colorscale='Viridis',
            colorbar=dict(title='Power (dBm)')
        )])
        
        fig3d.update_layout(
            title="Received Power (Pr)",
            scene=dict(
                xaxis_title='Distance (km)',
                yaxis_title='Frequency (MHz)',
                zaxis_title='Received Power (dBm)',
            ),
            margin=dict(l=40, r=40, b=40, t=80)
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("#### Variant 2: Received Power vs. Antenna Gains (Gt, Gr)")
        
        # Create 3D data
        gt_3d_dbi = np.linspace(max(-10, gt_dbi - 15), gt_dbi + 15, 40)
        gr_3d_dbi = np.linspace(max(-10, gr_dbi - 15), gr_dbi + 15, 40)
        
        GT_dbi, GR_dbi = np.meshgrid(gt_3d_dbi, gr_3d_dbi)
        GT_lin = dbi_to_linear(GT_dbi)
        GR_lin = dbi_to_linear(GR_dbi)
        
        PR_w_3D_gains = pt_w * GT_lin * GR_lin * (lambda_m / (4 * np.pi * dist_m))**2
        PR_dbm_3D_gains = watts_to_dbm(PR_w_3D_gains)

        fig3d_2 = go.Figure(data=[go.Surface(
            z=PR_dbm_3D_gains,
            x=gt_3d_dbi,
            y=gr_3d_dbi,
            colorscale='Plasma',
            colorbar=dict(title='Power (dBm)')
        )])
        
        fig3d_2.update_layout(
            title="Received Power (Pr) vs. Antenna Gains",
            scene=dict(
                xaxis_title='Transmitter Gain (dBi)',
                yaxis_title='Receiver Gain (dBi)',
                zaxis_title='Received Power (dBm)',
            ),
            margin=dict(l=40, r=40, b=40, t=80)
        )
        st.plotly_chart(fig3d_2, use_container_width=True)


# --- Page 2: Radar Range Equation Lab ---
def show_radar_lab():
    """Renders the Streamlit page for the Radar Range Equation."""
    st.title("🛰️ Radar Range Equation Lab")
    st.markdown("Explore the maximum range of a radar system.")

    with st.expander("🔬 Lab Notes: The Equation", expanded=True):
        st.latex(r'''
        R_{\text{max}} = \left[ \frac{P_t G^2 \lambda^2 \sigma}{(4\pi)^3 P_{\text{min}}} \right]^{1/4}
        ''')
        st.markdown(r"""
        Where:
        - $R_{\text{max}}$ = Maximum Radar Range
        - $P_t$ = Transmitted Power
        - $G$ = Antenna Gain (assumes $G_t = G_r$)
        - $\lambda$ = Wavelength ($c/f$)
        - $\sigma$ = Radar Cross Section (RCS) of the target
        - $P_{\text{min}}$ = Minimum Detectable Signal
        """)

    tab_calc, tab_2d, tab_3d = st.tabs(["🎛️ Inputs & Calculation", "📊 2D Plots", "📈 3D Plots"])

    with tab_calc:
        st.subheader("Input Parameters")
        
        # --- Fun Mode: Presets ---
        st.markdown("#### Fun Mode (Target Presets)")
        rcs_preset = st.selectbox("Choose a target (sets RCS):",
                                  ("Custom",
                                   "Bird (0.01 $m^2$)",
                                   "Stealth Aircraft (0.1 $m^2$)",
                                   "Person (1 $m^2$)",
                                   "Fighter Jet (5 $m^2$)",
                                   "Car (100 $m^2$)",
                                   "Ship (10,000 $m^2$)"))

        # Default values
        defaults = {
            "pt_kw": 100.0,
            "g_dbi": 35.0,
            "freq_mhz": 3000.0,
            "pmin_dbm": -100.0,
            "rcs_m2": 1.0
        }

        if rcs_preset != "Custom":
            # Extract RCS value from the string
            defaults["rcs_m2"] = float(rcs_preset.split('(')[1].split(' ')[0])

        # --- Lab Mode: Manual Inputs ---
        st.markdown("---")
        st.markdown("#### Lab Mode (Manual Inputs)")
        
        col1, col2 = st.columns(2)
        with col1:
            pt_kw = st.number_input("Transmitted Power (Pt) [kW]", 0.1, 10000.0, defaults["pt_kw"], 10.0, format="%.1f")
            g_dbi = st.slider("Antenna Gain (G) [dBi]", 0.0, 60.0, defaults["g_dbi"], 0.5)
        
        with col2:
            freq_mhz = st.number_input("Frequency (f) [MHz]", 1.0, 100000.0, defaults["freq_mhz"], 100.0, format="%.1f")
            pmin_dbm = st.slider("Min. Detectable Signal (Pmin) [dBm]", -150.0, -30.0, defaults["pmin_dbm"], 0.5)

        # RCS slider (logarithmic for better control)
        st.markdown("Target Radar Cross Section ($\sigma$) [$m^2$]")
        # Using a log-slider trick with select_slider
        rcs_options = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        try:
            default_rcs_index = rcs_options.index(defaults["rcs_m2"])
        except ValueError:
            # If default isn't in options, add it and sort
            rcs_options.append(defaults["rcs_m2"])
            rcs_options.sort()
            default_rcs_index = rcs_options.index(defaults["rcs_m2"])

        rcs_m2 = st.select_slider("", options=rcs_options, value=rcs_options[default_rcs_index])
        
        # --- Calculation ---
        pt_w = pt_kw * 1000
        g_lin = dbi_to_linear(g_dbi)
        freq_hz = freq_mhz * 1e6
        lambda_m = C / freq_hz
        pmin_w = dbm_to_watts(pmin_dbm)

        # Radar Range Equation
        numerator = pt_w * (g_lin**2) * (lambda_m**2) * rcs_m2
        denominator = ((4 * np.pi)**3) * pmin_w
        
        rmax_m = (numerator / denominator)**0.25
        rmax_km = rmax_m / 1000

        st.markdown("---")
        st.subheader("⚡ Results")
        st.metric("Maximum Detectable Range (R_max)", f"{rmax_km:.2f} km")
        
        st.markdown("---")
        st.subheader("📊 Advanced Analysis & Interpretation")

        # Interpretation
        st.markdown("#### Interpretation: The 4th Power Law")
        st.info(f"""
        Notice the $R^{{1/4}}$ relationship. This is the most critical concept in radar design.
        
        - To **double** the range (x2), you must increase transmit power by **16 times** ($2^4$)!
        - To **triple** the range (x3), you must increase power by **81 times** ($3^4$)!
        
        This shows why long-range radar is so difficult and power-hungry. The same law applies to antenna gain ($G^2 \propto R^4 \implies G \propto R^2$) and target RCS ($\sigma \propto R^4$).
        """)
        
        # Reverse Calculator
        st.markdown("#### Analysis: Required Power for a Target Range")
        target_range_km = st.number_input("Enter your desired range (km):", 0.1, rmax_km * 5 if rmax_km < 10000 else 50000.0, rmax_km, 1.0)
        target_range_m = target_range_km * 1000
        
        # Rearrange the equation for Pt
        # R_max^4 = (Pt * G^2 * lambda^2 * rcs) / ((4pi)^3 * Pmin)
        # Pt = (R_max^4 * (4pi)^3 * Pmin) / (G^2 * lambda^2 * rcs)
        
        pt_denom = (g_lin**2) * (lambda_m**2) * rcs_m2
        
        if pt_denom > 1e-30:
            pt_req_w = (target_range_m**4 * denominator) / pt_denom
            
            if pt_req_w < 1000:
                 st.metric(f"Required Power for {target_range_km} km", f"{pt_req_w:,.2f} W")
            elif pt_req_w < 1_000_000:
                 st.metric(f"Required Power for {target_range_km} km", f"{pt_req_w / 1000:,.2f} kW")
            elif pt_req_w < 1_000_000_000:
                 st.metric(f"Required Power for {target_range_km} km", f"{pt_req_w / 1_000_000:,.2f} MW")
            else:
                 st.metric(f"Required Power for {target_range_km} km", f"{pt_req_w / 1_000_000_000:,.2f} GW")
        else:
            st.error("Cannot calculate required power with zero gain or RCS.")
            

    with tab_2d:
        st.subheader("2D Visualizations")
        
        # Plot 1: Range vs. Transmit Power
        st.markdown("#### Range vs. Transmit Power")
        pt_range_kw = np.linspace(pt_kw / 10 if pt_kw > 1 else 0.1, pt_kw * 3, 200)
        pt_range_w = pt_range_kw * 1000
        num_range_pt = pt_range_w * (g_lin**2) * (lambda_m**2) * rcs_m2
        rmax_m_pt = (num_range_pt / denominator)**0.25
        rmax_km_pt = rmax_m_pt / 1000

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=pt_range_kw, y=rmax_km_pt, mode='lines', name='Max Range'))
        fig1.add_vline(x=pt_kw, line_dash="dash", line_color="red", annotation_text="Current Power")
        fig1.update_layout(title="Max Range as Transmit Power Changes",
                           xaxis_title="Transmit Power (kW)",
                           yaxis_title="Max Range (km)")
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Range vs. RCS
        st.markdown("#### Range vs. Target RCS (Log Scale)")
        rcs_range_m2_plot = np.logspace(np.log10(rcs_m2 / 100 if rcs_m2 > 0.01 else 0.0001), np.log10(rcs_m2 * 100 if rcs_m2 < 10000 else 1000000), 200)
        num_range_rcs = pt_w * (g_lin**2) * (lambda_m**2) * rcs_range_m2_plot
        rmax_m_rcs = (num_range_rcs / denominator)**0.25
        rmax_km_rcs = rmax_m_rcs / 1000

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rcs_range_m2_plot, y=rmax_km_rcs, mode='lines', name='Max Range'))
        fig2.add_vline(x=rcs_m2, line_dash="dash", line_color="red", annotation_text="Current RCS")
        fig2.update_layout(title="Max Range as Target RCS Changes",
                           xaxis_title="RCS ($\sigma$) [$m^2$] (Log Scale)",
                           yaxis_title="Max Range (km)",
                           xaxis_type="log")
        st.plotly_chart(fig2, use_container_width=True)

        # Plot 3: Range vs. Antenna Gain
        st.markdown("#### Max Range vs. Antenna Gain")
        g_range_dbi = np.linspace(max(0, g_dbi - 20), g_dbi + 20, 200)
        g_range_lin = dbi_to_linear(g_range_dbi)
        num_range_g = pt_w * (g_range_lin**2) * (lambda_m**2) * rcs_m2
        rmax_m_g = (num_range_g / denominator)**0.25
        rmax_km_g = rmax_m_g / 1000

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=g_range_dbi, y=rmax_km_g, mode='lines', name='Max Range'))
        fig3.add_vline(x=g_dbi, line_dash="dash", line_color="red", annotation_text="Current Gain")
        fig3.update_layout(title="Max Range as Antenna Gain (G) Changes",
                           xaxis_title="Antenna Gain (dBi)",
                           yaxis_title="Max Range (km)")
        st.plotly_chart(fig3, use_container_width=True)


    with tab_3d:
        st.subheader("3D Visualization")
        st.markdown("#### Variant 1: R_max vs. Transmit Power & RCS")

        # Create 3D data
        pt_3d_kw = np.linspace(pt_kw / 10 if pt_kw > 1 else 0.1, pt_kw * 3, 50)
        rcs_3d_m2 = np.logspace(np.log10(rcs_m2 / 100 if rcs_m2 > 0.01 else 0.0001), np.log10(rcs_m2 * 100), 50)
        
        # Create meshgrid
        PT_kw, RCS = np.meshgrid(pt_3d_kw, rcs_3d_m2)
        PT_w = PT_kw * 1000
        
        # Calculate Rmax in 3D
        NUM_3D = PT_w * (g_lin**2) * (lambda_m**2) * RCS
        RMAX_m_3D = (NUM_3D / denominator)**0.25
        RMAX_km_3D = RMAX_m_3D / 1000

        fig3d = go.Figure(data=[go.Surface(
            z=RMAX_km_3D, 
            x=PT_kw, 
            y=np.log10(RCS),  # Use log scale for y-axis ticks
            colorscale='Inferno',
            colorbar=dict(title='Max Range (km)')
        )])
        
        fig3d.update_layout(
            title="Maximum Radar Range",
            scene=dict(
                xaxis_title='Transmit Power (kW)',
                yaxis_title='RCS [$m^2$] (Log Scale)',
                zaxis_title='Max Range (km)',
                # Format y-axis ticks to show linear values from log
                yaxis = dict(
                    tickvals = np.log10(rcs_3d_m2[::10]), # subset of ticks
                    ticktext = [f"{v:.2f}" for v in rcs_3d_m2[::10]]
                )
            ),
            margin=dict(l=40, r=40, b=40, t=80)
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("#### Variant 2: R_max vs. Frequency & Antenna Gain")
        
        # Create 3D data
        freq_3d_mhz = np.linspace(max(1, freq_mhz / 4), freq_mhz * 3, 40)
        g_3d_dbi = np.linspace(max(0, g_dbi - 20), g_dbi + 20, 40)
        
        F_mhz, G_dbi = np.meshgrid(freq_3d_mhz, g_3d_dbi)
        
        F_hz = F_mhz * 1e6
        LAMBDA_m_3d = C / F_hz
        G_lin_3d = dbi_to_linear(G_dbi)
        
        NUM_3D_fg = pt_w * (G_lin_3d**2) * (LAMBDA_m_3d**2) * rcs_m2
        RMAX_m_3D_fg = (NUM_3D_fg / denominator)**0.25
        RMAX_km_3D_fg = RMAX_m_3D_fg / 1000
        
        fig3d_2 = go.Figure(data=[go.Surface(
            z=RMAX_km_3D_fg,
            x=freq_3d_mhz,
            y=g_3d_dbi,
            colorscale='Cividis',
            colorbar=dict(title='Max Range (km)')
        )])
        
        fig3d_2.update_layout(
            title="Max Range vs. Frequency & Antenna Gain",
            scene=dict(
                xaxis_title='Frequency (MHz)',
                yaxis_title='Antenna Gain (dBi)',
                zaxis_title='Max Range (km)',
            ),
            margin=dict(l=40, r=40, b=40, t=80)
        )
        st.plotly_chart(fig3d_2, use_container_width=True)


# --- Main App Logic (Navigation) ---
def main():
    st.set_page_config(page_title="RF Virtual Lab", page_icon="🔬", layout="wide")
    
    st.sidebar.title("🔬 RF Lab Navigator")
    st.sidebar.markdown("Select a virtual lab to begin your experiment.")
    
    page = st.sidebar.radio("Choose your lab:", 
                            ("Home", 
                             "📡 Friis Transmission Lab", 
                             "🛰️ Radar Range Lab"))
    st.sidebar.markdown("---")
    st.sidebar.info("Built with Streamlit & Plotly")

    if page == "Home":
        st.title("Welcome to the Virtual RF Lab! 🔬")
        st.markdown("""
        This application is an interactive, web-based laboratory for exploring
        two fundamental equations in radio frequency (RF) engineering.
        
        Use the **navigation on the left** to select a lab and start experimenting.
        
        ### 📡 Friis Transmission Lab
        - Calculate the power received by an antenna given a transmitter's properties
          and the distance between them.
        - **See a full link budget breakdown** and a plain-English interpretation of the link quality.
        - **Play with presets** like WiFi, Bluetooth, and satellite links.
        - Explore multiple 2D and 3D plots to see how parameters interact.
        
        ### 🛰️ Radar Range Lab
        - Calculate the maximum range a radar system can detect a target.
        - **Learn about the critical "4th Power Law"** and why radar is so power-hungry.
        - **Use the reverse calculator** to find the power needed for a desired detection range.
        - **Play with target presets** like a person, a car, or a stealth aircraft.
        - Visualize the design trade-offs with new 2D and 3D graphs.
        
        Enjoy your experiments!
        """)
    elif page == "📡 Friis Transmission Lab":
        show_friis_lab()
    elif page == "🛰️ Radar Range Lab":
        show_radar_lab()

if __name__ == "__main__":
    main()

