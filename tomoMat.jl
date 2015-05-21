module tomoMat

export makeCovMap

function makeCovMap(N::Integer, D, r0)
    covMaps = zeros(Float32, (2, N, N))

    ny = 1
    for ky = linrange(-1/D,1/D,N)
        nx = 1
        for kx = linrange(-1/D,1/D,N)
            k = sqrt(kx^2 + ky^2)

            #@printf("nx: %d, \tkx: %f, \tny: %d,\tky: %f\n", nx, kx, ny, ky)
            covMaps[1, nx, ny] = kx^2 * r0^(-5/3) * k^(-11/3)
            covMaps[2, nx, ny] = ky^2 * r0^(-5/3) * k^(-11/3)
            nx+=1
        end
        ny+=1
    end
    covMaps
end


function makeCovMap(N::Integer, D, r0, L0, l0)
    covMaps = zeros(Float32, (2, N, N))

    ny = 1
    for ky = linrange(-1/D,1/D,N)
        nx = 1
        for kx = linrange(-1/D,1/D,N)
            k = sqrt(kx^2 + ky^2)

            covMaps[1, nx, ny] = kx^2 * r0^(-5/3) * exp(-(k^2 * l0^2/5.92)) * ((k^2 + 2*pi*L0^(-2.))^(-11/6))
            covMaps[2, nx, ny] = ky^2 * r0^(-5/3) * exp(-(k^2 * l0^2/5.92)) * ((k^2 + 2*pi*L0^(-2.))^(-11/6))
            nx+=1
        end
        ny+=1
    end
    covMaps
end

#End module
end


