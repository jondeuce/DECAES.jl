"""
decay_curve = EPGdecaycurve_vTE(ETL, flip_angle, TE1, TE2, nTE1, T2, T1, refcon)

Computes the normalized echo decay curve for a MR spin echo sequence with
the given parameters.

ETL:        Echo train length (number of echos)
flip_angle: Angle of refocusing pulses (degrees)
TE1:        First Interecho time (seconds)
TE2:        second Interecho time (seconds)
nTE1:       number of echoes at first echo time
T2:         Transverse relaxation time (seconds)
T1:         Longitudinal relaxation time (seconds)
refcon:     Value of Refocusing Pulse Control Angle
"""
function EPGdecaycurve_vTE_work(T, ETL)
    error("Placeholder function --- not implemented")
end
EPGdecaycurve_vTE_work(ETL) = EPGdecaycurve_vTE_work(Float64, ETL)

EPGdecaycurve_vTE(ETL::Int, flip_angle::T, TE1::T, TE2::T, nTE1::Int, T2::T, T1::T, refcon::T) where {T} =
    EPGdecaycurve_vTE!(EPGdecaycurve_vTE_work(T, ETL), ETL, flip_angle, TE1, TE2, nTE1, T2, T1, refcon)

function EPGdecaycurve_vTE!(work, ETL::Int, flip_angle::T, TE1::T, TE2::T, nTE1::Int, T2::T, T1::T, refcon::T) where {T}
    error("Placeholder function --- not implemented")
end
