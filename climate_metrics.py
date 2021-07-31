"""
This module implements methods for computing climate metrics as described
by the International Governmental Panel on Climate Change."""


import numpy as np


# W m–2 ppbv-1
RADIATIVE_EFFICIENCY_ppbv = {"co2": 1.37e-5, "ch4": 3.63e-4, "n2o": 3.00e-3}
COEFFICIENT_WEIGHTS = np.array([0.2173, 0.2240, 0.2824, 0.2763])
TIME_SCALES = np.array([394.4, 36.54, 4.304])


def _get_GHG_lifetime(ghg):
    ghg_lifetimes = dict(
        ch4=12.4,
        n2o=121
    )
    return ghg_lifetimes[ghg]


def _ppbv_to_kg_conversion(ghg):
    """
    Convert the radiative efficiency from ppbv normalization to kg normalization.

    References
    --------------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. p. 8SM-15.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    # kg per kmol
    molecular_weight = {"co2": 44.01, "ch4": 16.04, "n2o": 44.013}

    total_mass_atmosphere = 5.1352e18  # kg
    mean_molecular_weight_air = 28.97  # kg per kmol
    molecular_weight_ghg = molecular_weight[ghg]
    mass_ratio = mean_molecular_weight_air/molecular_weight_ghg
    return mass_ratio * (1e9/total_mass_atmosphere)


def _get_radiative_efficiency_kg(ghg):
    """Get the radiative efficiency of a GHG in W m–2 kg–1.
    """
    ppv_to_kg = _ppbv_to_kg_conversion(ghg)
    return ppv_to_kg * RADIATIVE_EFFICIENCY_ppbv[ghg]


def CO2_irf(time_horizon):
    """The impulse response function of CO2.

    Parameters
    -----------
    time_horizon : int
        The time since the original CO2 emission occurred.

    References
    --------------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. Equation 8.SM.10
    https://www.ipcc.ch/report/ar5/wg1/
    """

    exponential_1 = np.exp(-time_horizon/TIME_SCALES[0])
    exponential_2 = np.exp(-time_horizon/TIME_SCALES[1])
    exponential_3 = np.exp(-time_horizon/TIME_SCALES[2])

    return (
        COEFFICIENT_WEIGHTS[0]
        + COEFFICIENT_WEIGHTS[1]*exponential_1
        + COEFFICIENT_WEIGHTS[2]*exponential_2
        + COEFFICIENT_WEIGHTS[3]*exponential_3
    )


def impulse_response_function(t, ghg):
    """The impulse response function for non-CO2/CH4 GHGs.

    References
    -----------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. Equation 8.SM.8.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    life_time = {"ch4": 12.4, "n2o": 121}
    if ghg.lower() == "co2":
        return CO2_irf(t)
    else:

        return np.exp(-t/life_time[ghg.lower()])


def radiative_forcing_per_kg(t, ghg):
    """Computes the radiative forcing at time `t` for a GHG emission at time 0.

    Parameters
    --------------
    t : array_like
        Time at which radiative forcing is computed.
    """
    if ghg.lower() == 'co2':
        radiative_efficiency = _get_radiative_efficiency_kg(ghg)

    elif ghg.lower() == 'ch4':
        radiative_efficiency = _get_radiative_efficiency_kg(ghg)
        radiative_efficiency *= _scaled_radiative_efficiency_from_O3_and_H2O()

    elif ghg.lower() == 'n2o':
        radiative_efficiency = _N2O_radiative_efficiency_after_methane_adjustment()

    return radiative_efficiency * impulse_response_function(t, ghg)


def radiative_forcing_from_emissions_scenario(
        time_horizon,
        emissions,
        ghg,
        step_size,
        mode='full'):
    """
    Parameters
    ---------------
    time_horizon : int
        Time period over which radiative forcing is computed for `emissions`.
    emissions : array_like
        GHG emissions (kg) at each time step.
    ghg : str
    step_size : float
        step_size for emissions to create a time index (`t`)
    mode : {'full', 'valid'}, optional
        mode passed to np.convolve.
        'full':
            Use full to get the temporal change in radiative forcing
            from 0-`time_horizon`.
            Output shape is `len(time_horizon) + len(emissions) + 1`.
        'same':
            Use to get the radiative forcing at `time_horizon`.
            Output shape is `max(len(time_horizon), len(emissions))`.

    Returns
    ---------------
    ndarray
    """
    t = np.arange(0, time_horizon+step_size, step_size)
    assert len(t) >= len(emissions)
    rf = radiative_forcing_per_kg(t, ghg)
    steps = int(time_horizon/step_size)
    return _convolve_metric(steps, emissions, rf, mode)


def _convolve_metric(steps, emissions, metric, mode):
    if mode == 'full':
        return np.convolve(emissions, metric, mode=mode)[0:steps+1]
    elif mode == 'valid':
        return np.convolve(emissions, metric, mode=mode)
    else:
        raise ValueError(f'Received invalid mode value: {mode}')


def AGWP_CO2(t):
    radiative_efficiency = _get_radiative_efficiency_kg("co2")
    exponential_1 = 1 - np.exp(-t/TIME_SCALES[0])
    exponential_2 = 1 - np.exp(-t/TIME_SCALES[1])
    exponential_3 = 1 - np.exp(-t/TIME_SCALES[2])
    cumulative_concentration = (
        COEFFICIENT_WEIGHTS[0]*t
        + COEFFICIENT_WEIGHTS[1]*TIME_SCALES[0]*exponential_1
        + COEFFICIENT_WEIGHTS[2]*TIME_SCALES[1]*exponential_2
        + COEFFICIENT_WEIGHTS[3]*TIME_SCALES[2]*exponential_3
        )

    return radiative_efficiency * cumulative_concentration


def _scaled_radiative_efficiency_from_O3_and_H2O():
    indirect_O3 = 0.5
    indirect_H2O = 0.15
    return 1 + indirect_O3 + indirect_H2O


def AGWP_CH4_no_CO2(t):
    """
    Parameters
    -----------
    t : int
        
    Note
    ------
    Does not include indirect effects from CO2 as a result of CH4 conversion to CO2.
    """
    radiative_efficiency = _get_radiative_efficiency_kg("ch4")
    methane_adjustments = _scaled_radiative_efficiency_from_O3_and_H2O()

    return (
        radiative_efficiency
        * methane_adjustments
        * _get_GHG_lifetime('ch4')
        * (1 - impulse_response_function(t, 'ch4'))
    )


def _N2O_radiative_efficiency_after_methane_adjustment():
    indirect_effect_of_N2O_on_CH4 = 0.36
    methane_adjustments = _scaled_radiative_efficiency_from_O3_and_H2O()
    radiative_efficiency_CH4_ppbv = RADIATIVE_EFFICIENCY_ppbv['ch4']
    radiative_efficiency_N2O_ppbv = RADIATIVE_EFFICIENCY_ppbv['n2o']
    radiative_efficiency_methane_adjustment = (
        indirect_effect_of_N2O_on_CH4
        * methane_adjustments
        * (radiative_efficiency_CH4_ppbv / radiative_efficiency_N2O_ppbv)
    )
    radiative_efficiency_N2O = _get_radiative_efficiency_kg("n2o")

    net_radiative_efficiency = (
        radiative_efficiency_N2O
        * (1 - radiative_efficiency_methane_adjustment)
    )
    return net_radiative_efficiency


def AGWP_N2O(t):
    net_radiative_efficiency = _N2O_radiative_efficiency_after_methane_adjustment()
    lifetime_N2O = _get_GHG_lifetime('n2o')
    irf_N2O = impulse_response_function(t, 'n2o')

    return (
        net_radiative_efficiency
        * lifetime_N2O
        * (1 - irf_N2O)
    )


def AGWP(ghg, t):
    if ghg.lower() == 'co2':
        return AGWP_CO2(t)
    elif ghg.lower() == 'ch4':
        return AGWP_CH4_no_CO2(t)
    elif ghg.lower() == 'n2o':
        return AGWP_N2O(t)
    else:
        raise NotImplementedError(f'AGWP methods have not been implemented for {ghg}')


def dynamic_AGWP(time_horizon, net_emissions, ghg, step_size, mode='valid'):
    """
    """

    t = np.arange(0, time_horizon+step_size, step_size)

    # To compute the convolution over t, net_emissions needs to be as long as t.
    if len(t) < len(net_emissions):
        raise ValueError(
            f"Expected time axis to always be larger than net_emissions \
                {net_emissions.shape}, {t.shape}.")

    AGWP_GHG = AGWP(ghg, t)
    steps = int(time_horizon/step_size)
    return _convolve_metric(steps, net_emissions, AGWP_GHG, mode=mode)


def GWP(time_horizon,
        emissions,
        ghg,
        step_size=1,
        annual=False):
    """Compute global warming potential of emissions.

    Parameters
    -----------
    time_horizon : int
    emissions : int or ndarray
        If emissions is an int, the emission is assumed to 
            occur at time=0.
    ghg : str {'CO2', 'CH4', 'N2O'}, optional
        Type of GHG emission in `emissions`.
    step_size : float or int
        Step size of `emissions` in years.
    annual : bool
        If `True`, returns annual GWP over the `time_horizon`. If `False`,
            returns the single value at `time_horizon`.
    """

    if type(emissions) is int:
        empty_array = np.zeros(time_horizon+step_size)
        empty_array[0] = emissions
        emissions = empty_array

    result = dynamic_GWP(
        time_horizon,
        emissions,
        ghg,
        step_size,
        mode='full')

    if annual:
        return result
    else:
        return result[time_horizon * int(1/step_size)]


def dynamic_GWP(
        time_horizon,
        net_emissions,
        ghg,
        step_size=0.1,
        mode='valid'):
    """Computes CO2 equivalent radiative forcing for net_emissions.

    Notes
    ------------
    If step_size < 1, the sum of the net_emissions vector must be equal
    to total emissions. So if a probability density function were used
    to simulate net_emissions over time, net_emissions would first have
    to be weighted by step_size before being passed to this function.

    Global Warming Potential is defined as the cumulative radiative forcing
    of :math:`GHG_x` emitted in year = 0 over a given time-horizon
    (:math:`t`):

    .. math:
        GWP(t) = \\frac{cumulativeRadiativeForcingGHG\\_x(t)}
                    {cumulativeRadiativeForcing\\_CO2(t)}

    Dynamic GWP ([1]_, [2]_ [3]_, [4]_) is the cumulative radiative forcing due
    to annual emissions (:math:`t'`) of :math:`GHG_x` over a give time-horizon
    (:math:`t`) which can be expressed as:

    .. math:
        dynamicGWP_x(t, t')
                    = {\\mathbf{emission_x}(t')}\\cdot{\\mathbf{GWP_x}(t-t')}
                    = \\sum_{t'}{\\mathbf{emission_x}(t'){\\mathbf{GWP_x}(t-t')}}
                    = \frac{
                    \\sum_{t'}{cumulativeRadiativeForcingGHG_x(t-t')}}
                    {cumulativeRadiativeForcing_{CO2}(t)}


    Parameters
    ---------------
    time_horizon : int
        The time horizon over which radiative forcing is computed.
    net_emissions : np.array
        Annual emissions/removals.
    is_unit_impulse : bool
        Specifies whether net_emissions is a scipy.signal.unit_impulse


    Notes
    ---------------
    net_emissions should not contain both unit_impulse and pdf emission
    distributions due to numerical integration issues.  After numerical
    integration, the output is re-normalized using the step_size when
    net_emissions is_unit_impulse=False. When is_unit_impulse is true,
    this normalization is not required.

    TODO
    -----------------
    update the method to take an arbitrary AGWP_GHG method

    References
    --------------
    .. [1] Fearnside et al. 2000.  https://link.springer.com/article/10.1023/A:1009625122628  # noqa: E501
    .. [2] Moura Costa et al. 2000.  https://link.springer.com/article/10.1023/A:1009697625521  # noqa: E501
    .. [3] Levassuer et al. 2010.  https://pubs.acs.org/doi/10.1021/es9030003
    .. [4] Cherubini et al. 2011.  https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1757-1707.2011.01102.x  # noqa: E501


    """
    dynamic_AGWP_GHG = dynamic_AGWP(time_horizon, net_emissions, ghg, step_size, mode)
    # A step of 0.1 is recommended to reduce the integration error
    # AGWP for each time step
    dynamic_GWP_t = dynamic_AGWP_GHG / AGWP_CO2(100)

    return dynamic_GWP_t


# Short-term and long-term temperature response
# (Kelvin per (Watt per m2)) to radiative forcing
TEMPERATURE_RESPONSE_COEFFICIENTS = [0.631, 0.429]
# Temporal scaling factors (years)
TEMPORAL_WEIGHTS = [8.4, 409.5]


def AGTP_CO2(t):
    """

    References
    ------------
    1. 8.SM.15 in https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
    """
    radiative_efficiency = _get_radiative_efficiency_kg("co2")

    temperature_response = 0
    for j in range(2):
        short_term_temperature_response = COEFFICIENT_WEIGHTS[0] \
            * TEMPERATURE_RESPONSE_COEFFICIENTS[j]
        temporal_weight_1 = np.exp(-t/TEMPORAL_WEIGHTS[j])
        weighted_short_term_temperature_response = short_term_temperature_response \
            * (1 - temporal_weight_1)

        weighted_long_term_temperature_response = 0
        for i in range(3):
            temporal_weight_2_linear = TIME_SCALES[i] \
                / (TIME_SCALES[i] - TEMPORAL_WEIGHTS[j])
            long_term_temperature_response = COEFFICIENT_WEIGHTS[i+1] \
                * TEMPERATURE_RESPONSE_COEFFICIENTS[j]
            long_term_temperature_response = long_term_temperature_response \
                * temporal_weight_2_linear
            temporal_weight_2_exponential = np.exp(-t/TIME_SCALES[i])
            weighted_long_term_temperature_response += (
                long_term_temperature_response
                * (temporal_weight_2_exponential - temporal_weight_1)
            )

        temperature_response += (
            weighted_short_term_temperature_response
            + weighted_long_term_temperature_response
        )
    return radiative_efficiency * temperature_response


def AGTP_non_CO2(t, ghg):
    radiative_efficiency = _get_radiative_efficiency_kg(ghg)
    ghg_lifetime = _get_GHG_lifetime(ghg)

    temperature_response = 0
    for i in range(2):
        temporal_weight_linear = ghg_lifetime / (ghg_lifetime - TEMPORAL_WEIGHTS[i])
        temperature_response_coefficient = TEMPERATURE_RESPONSE_COEFFICIENTS[i]
        irf_GHG = impulse_response_function(t, ghg)
        delayed_temperature_response = np.exp(-t/TEMPORAL_WEIGHTS[i])
        temperature_response += (
            temporal_weight_linear
            * temperature_response_coefficient
            * (irf_GHG - delayed_temperature_response)
        )

    if ghg.lower() == 'ch4':
        methane_adjustments = _scaled_radiative_efficiency_from_O3_and_H2O()
        return (
            methane_adjustments
            * radiative_efficiency
            * temperature_response
        )
    elif ghg.lower() == 'n2o':
        net_radiative_efficiency = _N2O_radiative_efficiency_after_methane_adjustment()
        return net_radiative_efficiency * temperature_response
    else:
        return radiative_efficiency * temperature_response


def AGTP(t, ghg):
    if ghg.lower() == 'co2':
        return AGTP_CO2(t)
    else:
        return AGTP_non_CO2(t, ghg)


def GTP(t, ghg) -> float:
    """
    Computes the global temperature change potential for 1 kg of `ghg`.

    Parameters
    t : int
        The time at which GTP is computed.
    ghg : str
        The ghg for which GTP is computed.

    """
    return AGTP(t, ghg)/AGTP_CO2(t)


def dynamic_GTP(time_horizon, emissions, ghg, step_size, mode='valid'):
    """
    Global average surface temperature change due to an `emissions` vector.

    Parameters
    ------------------
    time_horizon : int
        The time over which the temperature response is computed.
    ghg : str
    emissions : ndarray
        Emissions in kg of a `ghg`.
    step_size : float or int
        The step size used to generate the time axis.
    mode : {'full' or 'valid'}, optional
        'full':
            This provides the full temporal profile of the temperature response
            over the time 0-time_horizon.
        'valid':
            This provides the temperature response from the emission vector
            at `time_horizon`.


    References
        .. [1] Equation 8.1 in https://www.ipcc.ch/site/assets/uploads/2018/02/WG1AR5_Chapter08_FINAL.pdf  # noqa: E501
    """
    t = np.arange(0, time_horizon+step_size, step_size)
    AGTP_GHG = AGTP(t, ghg)
    if len(t) < len(emissions):
        raise ValueError("Expected time vector to be longer than the emissions vector")
    # return np.dot(emissions, np.flip(AGTP(ghg, t)))
    steps = int(time_horizon/step_size)
    return _convolve_metric(steps, emissions, AGTP_GHG, mode)
