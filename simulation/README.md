\# RetrofitIQ HVAC Simulation



\## Current status



The OpenStudio + EnergyPlus simulation pipeline is currently being debugged.



\### Completed



\- Building model created

\- 2 thermal zones

\- Baseline constant-volume AHU created

\- DX cooling coil and constant-volume fan added

\- Mumbai weather data selected

\- Indian Metro CSV converted to EPW

\- 8,760 hourly weather records

\- Weather attached to OpenStudio model

\- Dew-point calculation corrected from temperature + relative humidity



\## Current blocker



EnergyPlus 25.2.0 terminates with:



GetNextEnvironment: Runperiod \[mm/dd] (Start=01/01,End=12/31) requested not within Data Period(s) from Weather File



The relevant weather file is:



simulation/weather/Mumbai\_indian\_metro.epw



The simulation is launched using:



simulation/run\_mumbai.osw



\## Goal



Get a clean baseline HVAC energy simulation running first.



After that:



1\. Validate baseline energy consumption

2\. Create retrofit scenarios such as AHU VFD / smart controls

3\. Run retrofit simulations

4\. Compare baseline vs retrofit energy use

5\. Estimate energy and CO2 savings

6\. Integrate the results into RetrofitIQ

