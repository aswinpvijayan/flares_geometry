module calc_zlos
using PyCall
np = pyimport("numpy")

# function getLOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins)
#
#
#     n = round(Int, length(s_cood)/3)
#     Z_los_SD = zeros(Float64, n)
#     #Fixing the observer direction as z-axis. Use make_faceon() for changing the
#     #particle orientation to face-on
#     xdir, ydir, zdir = 1, 2, 3
#     for ii in 1:n
#
#         thisspos = s_cood[ii,:]
#         ok = @views @. g_cood[:,zdir].>thisspos[zdir]
#         m = sum(ok)
#         # thisgpos = Array{Float64}(undef, (m,3))
#         # thisgsml = Array{Float64}(undef, m)
#         # thisgZ = Array{Float64}(undef, m)
#         # thisgmass = Array{Float64}(undef, m)
#         # boverh = Array{Float64}(undef, m)
#
#
#         thisgpos = @views @. g_cood[ok,:]
#         thisgsml = g_sml[ok]
#         thisgZ = g_Z[ok]
#         thisgmass = g_mass[ok]
#         x = @views @. thisgpos[:,xdir].-thisspos[xdir]
#         y = @views @. thisgpos[:,ydir].-thisspos[ydir]
#
#         boverh = sqrt.(x.*x + y.*y)./thisgsml
#
#         ok = boverh.<=1.
#         kernel_vals = [lkernel[round(Int, kbins*ll + 1)] for ll in boverh[ok]]
#
#         Z_los_SD[ii] = sum(@views @. (thisgmass[ok].*thisgZ[ok]./(thisgsml[ok].*thisgsml[ok])).*kernel_vals) #in units of Msun/Mpc^2
#
#     end
#
#     return Z_los_SD
#
# end
#
# end

function getLOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins)


    n = round(Int, length(s_cood)/3)
    Z_los_SD = zeros(Float64, n)
    #Fixing the observer direction as z-axis. Use make_faceon() for changing the
    #particle orientation to face-on
    xdir, ydir, zdir = 1, 2, 3
    for ii in 1:n

        thisspos = s_cood[ii,:]
        ok = @views @. g_cood[:,zdir].>thisspos[zdir]

        thisgpos = @views @. g_cood[ok,:]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = @views @. thisgpos[:,xdir].-thisspos[xdir]
        y = @views @. thisgpos[:,ydir].-thisspos[ydir]

        boverh = sqrt.(x.*x + y.*y)./thisgsml

        ok = boverh.<=1.
        kernel_vals = [lkernel[round(Int, kbins*ll + 1)] for ll in boverh[ok]]

        Z_los_SD[ii] = sum(@views @. (thisgmass[ok].*thisgZ[ok]./(thisgsml[ok].*thisgsml[ok])).*kernel_vals) #in units of Msun/Mpc^2


    end

       return Z_los_SD

end

end
