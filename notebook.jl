### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7112c7f2-a35a-11ef-0005-b77b37b5bb2c
using ImageShow, ImageIO, Plots, FFTW, CUDA,NDTools, PlutoUI, IndexFunArrays, EllipsisNotation,  FunctionZeros, SpecialFunctions, FileIO

# ╔═╡ 55fc243d-b162-43fa-b755-846ab413d530
using PlutoTeachingTools

# ╔═╡ 1bc50357-ec6a-404a-a2b9-b8d31b153475
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!


However, the simulations take quite some minutes, so don't be impatient :)
"

# ╔═╡ 15fe405d-a554-44b4-a527-45ddcbcdfad8
ChooseDisplayMode()

# ╔═╡ e0ad01cc-622d-4505-ba8b-29b504f92300
begin
	FFTW.forget_wisdom()
	FFTW.set_num_threads(8)
end

# ╔═╡ 526365f7-9a74-4c7c-9d9e-95fd7de1b549
TableOfContents()

# ╔═╡ 323c620a-0652-41e6-9190-8d548cba3a1c
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ f4fa6dab-d64f-46be-9f13-6cccbf50400a
#var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 00b13143-d9bd-402f-858c-0270a012a61f
#togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 6ce9a70c-0162-480a-9177-2715ff2f963a
#ImageShow.simshow(x::CuArray, kwargs...) = simshow(Array(x), kwargs...) 

# ╔═╡ 57387f24-15a4-4ea8-867e-e7d524c310b4
md"# Limits and Possibilities of the Multi Slice Propagation

[Felix Wechsler](https://www.felixwechsler.science)

PhD Student at the Laboratory of Applied Photonics Devices (LAPD)

Supervised by Christophe Moser

Presentation available: [go.epfl.ch/multi_slice](https://go.epfl.ch/multi_slice)
"

# ╔═╡ 443840db-b308-4202-b47d-7a96779f6648
load("qrcode.png")

# ╔═╡ d09a7268-e3f6-4ced-8cca-24b718895c19
md"""
# 1. Angular Spectrum Propagation

It is a convolutional operation.
It propagates a field with size $L$ over a distance $z$.

The output field size is $L$.

$$\psi_{z}=\mathcal{F}^{-1}\left[\mathcal{F}\left[\psi_{0}\right] \cdot H_{\textrm{AS}}\right],$$

$$H_{\textrm{AS}}\left(f_{x},f_{y}\right)=\exp\left(i2\pi z\sqrt{\frac{n^2}{\lambda^{2}}-f_{x}^{2}-f_{y}^{2}}\right)$$


The different variables $\psi_0$ input field, $f_x = \frac{k_x}{2\pi}$ spatial frequency, $\mathcal F$ Fourier Transform, $n$ refractive index in medium.


It is a solution to the homogenous Helmholtz equation in an isotropic medium.










"""

# ╔═╡ 5c16ec60-8531-40a6-9dde-8997b13f91e2
begin
	box = zeros((512,))
	box[200:300] .= 1
end

# ╔═╡ bcf50548-670b-43dd-8d10-0763aeac423b
function angular_spectrum(field, λ, L, z, n=1)
	ΔL = L / length(field)
	fₓ = fftfreq(length(field), inv(ΔL))
	H = @. exp(1im * 2π * z * √(Complex(n^2 / λ^2 - fₓ^2)))
	return ifft(fft(field) .* H)
end

# ╔═╡ 069ea1ca-d759-4fde-addb-ebf8cefd1596
x_box = range(-20e-6, 20e-6, 513)[begin:end-1];

# ╔═╡ 377246ab-19ba-4636-b527-83deac269574
@bind z_dist Slider(range(0, 1000e-6, 100), show_value=true)

# ╔═╡ 6dcc011d-6d91-401f-b4ef-f78ddce62398
box_propagated = angular_spectrum(box, 633e-9, 40e-6, z_dist, 1.0);

# ╔═╡ 24414435-617b-4fa5-a500-850ec67da0dc
box_propagated_13 = angular_spectrum(box, 633e-9, 40e-6, z_dist, 1.3);

# ╔═╡ 2d4d4582-65cf-48fa-b369-a808627feb6a
begin
	plot(x_box * 1e6, abs2.(box), label="input", xlabel="z in μm")
	plot!(x_box * 1e6, abs2.(box_propagated), label="propagated, n=1")
	plot!(x_box * 1e6, abs2.(box_propagated_13), label="propagated, n=1.3")
end

# ╔═╡ 6d3b05ba-c1c9-475b-82c8-6ffd1175f161
heatmap(range(0, 200e-6, 200) * 1e6, x_box * 1e6, abs2.(hcat((angular_spectrum(box, 633e-9, 40e-6, z) for z in range(0, 200e-6, 200))...)).^0.6, xlabel="z in μm", ylabel="x in μm")

# ╔═╡ d259a222-bdd1-4af1-8a32-be942551dc77


# ╔═╡ 185ac123-226f-443c-9882-c758677c120a
md"""


# 2. Thin Element Approximation

!!! warn "References"
	- J. M. Cowley and A. F. Moodie, “The Scattering of Electrons by Atoms and Crystals. I. A New Theoretical Approach,” Acta Cryst. 10 (1957)
	- K. Li, M. Wojcik, and C. Jacobsen, “Multislice does it all—calculating the performance of nanofocusing X-ray optics,” Opt. Express 25, 1831 (2017)}

If an optical field propagates through a thin medium with non-uniform refractive index, its effect on the beam can described as a multiplication.

For example, a thin lens with focal length $f$ has a refractive index profile of:

$$T_L(x,y) = \exp\left(- \frac{i\cdot k}{2\cdot f}\cdot (x^2 + y^2) \right)$$

So the electrical field directly after the lens $U_{+}(x,y)$ is simply:

$$U_{+}(x,y) = T_L(x,y) \cdot (U_{-}(x,y))$$

To observe the effect along the optical axis, we just propagate with the Angular Spectrum method.

"""

# ╔═╡ 7b6a90ce-c32d-495d-b642-2f5baa7c3e05
md"""
## Test your implementation

Always test your implementation!

Programming is hard, programming physics even harder.


In this case we put the thin phase mask of a lens, and check that the beam gets focused at this spot.
"""

# ╔═╡ 5bc6b3ae-29d5-4df7-8f35-2154ca0ef34b
begin
	box_2 = zeros((512,));
	box_2[50:512-50] .= 1
end;

# ╔═╡ 47d0d3d3-83f5-4263-af38-b21a20714744
x_box_2 = range(-40e-6, 40e-6, 513)[begin:end-1];

# ╔═╡ 5503ca5c-114e-434b-ab69-37d50e09bfea
focal_length = 150e-6

# ╔═╡ ac6dcc8a-e940-490b-b697-c61bcde333ec
box_2_lens = @. exp(-1im * 2π/ 633e-9 / 2 / focal_length * x_box_2^2) * box_2;

# ╔═╡ 6f7ad76b-a21c-4413-97e8-6a3b33c13a00
heatmap(range(0, 200e-6, 200) * 1e6, x_box_2 * 1e6, abs2.(hcat((angular_spectrum(box_2_lens, 633e-9, 80e-6, z) for z in range(0, 200e-6, 200))...)).^0.5, xlabel="z in μm", ylabel="x in μm")

# ╔═╡ 00509bfa-80af-4a4c-b246-a211d0ff1ca1


# ╔═╡ 9bdb1c18-12d6-45c4-8048-597c1926e056


# ╔═╡ 3f76ed4a-2c67-4d16-b6ba-a3388dd3a0c4
md"""
# 3. Multi Slice Propagation
So now we can apply this idea to a thicker volume by slicing the volume and apply the thin element approximation to each slice.
Between the slices we use free space propagation.

"""

# ╔═╡ 622af1b1-be09-404d-a566-0954c0ca486f
load("figures/1.png")

# ╔═╡ 209d08ae-168e-48e0-a4b7-6f842c291a73
md"## 1. Generate a Ball lens
Simply by binarizing the pixel values to the refractive index in the medium.
"

# ╔═╡ 8e798c1d-9386-4446-bb12-11d0a7de0277
load("figures/2.png")

# ╔═╡ 1397b1f7-599b-4df5-ad4d-8f626e527a26
N1 = 1024

# ╔═╡ 97f4e89f-b946-41a8-bfa8-b5325580e143
L1 = 60f-6

# ╔═╡ bdae79e7-5976-4a2e-b3da-be809d8de95f
Nz1 = 512

# ╔═╡ fe605fa6-b394-4531-8fff-93d7e755e8bb
n_air = 1.0f0

# ╔═╡ ca49b92f-b376-47a0-ab7b-b390eaa15bdf
n_glass = 1.5f0

# ╔═╡ b796d8c5-dba2-4b5a-9914-80566a5e004d
y1 = (range(-L1/2, L1/2, N1 + 1)[begin:end-1]);

# ╔═╡ 28e3d7bc-df79-4618-b723-9766706e1726
x1 = y1';

# ╔═╡ ed9fce6a-416c-41a5-a3df-cbd380dee960
z1 = reshape((range(-35f-6, 10f-6, Nz1 + 1)[begin:end-1]), 1, 1, Nz1);

# ╔═╡ 94fe19f5-dea0-48aa-9b64-e38961ea1176
lens1 = ((x1.^2 .+ y1.^2 .+ (z1 .+ 15f-6).^2) .<= (15f-6)^2) * (n_glass .- n_air) .+ n_air;

# ╔═╡ 4a117563-1ec8-4daf-b298-9e648e0e2990
md"## 2. Experimental Results with Gaussian Beam"

# ╔═╡ 589bf363-91b6-4381-902b-092da1e8c96e
w1 = 15f-6

# ╔═╡ ddf01f2c-8b7e-459c-86c5-f01f1c6aaa3c
beam = ComplexF32.(exp.(-(x1.^2 .+ y1.^2) ./ (w1^2)));

# ╔═╡ 36537308-0263-4555-9096-a90786849554
λ = 633f-9

# ╔═╡ 0537bafa-e13d-4ff1-92bc-6bbc35baa5d2
heatmap(x1[:] * 1e6, y1[:] * 1e6, Array(beam) .|> abs, xlabel="y in μm", ylabel="x in μm")

# ╔═╡ dc9b64ab-f619-4837-a6de-6fdd345a4d89
md"## 3. Multi Slice with Ball Lens"

# ╔═╡ ba492d7e-bf99-43b5-88e2-b6c23b24c89d
md"
The focal length of the ball lens can be calculated with the thick lensmaker equation

$$
\frac{1}{f} = (n-1) \left(\frac1{R_1} - \frac{1}{R_2} + \frac{2(n-1)\cdot d}{R_1 \cdot R_2 \cdot n} \right)$$

where $n$ is the refractive index, $R_i$ the radius of curvature of the two surfaces and $d$ the diameter.

In this case the focal length is $25\mathrm{\mu m}$.

"

# ╔═╡ eb06dc36-0f3a-4841-b7e9-31bd8f904cf7
function f(R₁, R₂, n, d=(abs(R₁) + abs(R₂))) 
	inv((n-1) *(1 / R₁ - 1 / R₂ + (n-1) * d / (n * R₁ * R₂)))
end

# ╔═╡ 97b68214-e2dc-4805-87a7-44526ebe5715
f(15e-6, -15e-6, 1.5)

# ╔═╡ d6125cbc-1def-4b8d-b4e1-1275ea304843
md"## 4. Generate a Smooth Boundary Lens
In the smooth lens, we calculate for pixels on the border of the lens, what the average refractive index would be.

This results in a more smooth lens surface.
"

# ╔═╡ 3c8b36b0-c777-4b03-b2d6-0cfa19e6b40c
load("figures/3.png")

# ╔═╡ 25c26f57-c5a4-4137-a8e2-407cbc9c7dbe
"""
	make_smooth_lens(_x, _y, _z, n_glass, n_air, N=50, radius=15f-6, offset=15f-6))

This returns an 3D array with the refractive index distribution of a spherical ball lens with refractiven index `n_glass` and surrounding medium `n_air`.
At the voxels which lie on the intersection of air and lens, we integrate in a Monte Carlo fashion the average refractive index.


# Optional Arguments
* `N=50` is the amount of Monte Carlo samples for the intersection voxels
* `radius` is the lens radius
* `offset` is the `z` center offset of the lens
"""
function make_smooth_lens(_x, _y, _z, n_glass, n_air, N=50, radius=15f-6, offset=15f-6)
	x = Array(_x)
	y = Array(_y)
	z = Array(_z)

	
	lens = Array(0 .* x .* y .* z)

	Δx = abs.(x[1] - x[2])	
	Δy = abs.(y[1] - y[2])
	Δz = abs.(z[1] - z[2])

	radius_squared = radius^2

	r() = rand(range(-0.5, 0.5, 1000)) 
	Threads.@threads for iz in 1:length(z)
		zp_s = (z[iz] + offset)^2
		for ix in 1:length(x)
			xp_s = x[ix]^2
			for iy in 1:length(y)
				counter = 0
				yp_s = y[iy]^2
				r1_squared = (xp_s + yp_s + zp_s)

				if (radius - (Δx + Δy + Δz))^2 < r1_squared < (radius + (Δx + Δy + Δz))^2 
					for _ in 1:N
						xp1 = x[ix] + r() * Δx
						yp1 = y[iy] + r() * Δy
						zp1 = z[iz] + r() * Δz
						counter += (xp1^2 + yp1^2 + (zp1 + radius)^2) <= radius_squared
					end
					lens[iy, ix, iz] = n_air + Float32(counter / N) * (n_glass - n_air)
				else
					if r1_squared > radius_squared
						lens[iy, ix, iz] = n_air
					else
						lens[iy, ix, iz] = n_glass
					end
				end
				
			end
		end
	end
	return lens
end

# ╔═╡ c2aac6a8-8daa-4cac-adff-e7dec47992b2
@time lens_smooth = make_smooth_lens(x1, y1, z1, n_glass, n_air);

# ╔═╡ c22b7856-ec84-44b6-91ed-3fcdf7653478
@bind ix33 PlutoUI.Slider(1:size(lens_smooth, 2), default=size(lens_smooth, 1)÷2 + 1)

# ╔═╡ 0f7e509f-804a-45f2-8149-792bb2583d16
heatmap(Array(z1[:]) * 1e6, Array(x1[:]) * 1e6, lens1[ix33, :, :], title="Hard Lens - x slice at $(round(x1[ix33] * 1e6, digits=2))μm", xlabel="z in μm", ylabel="x in μm")

# ╔═╡ f4d61401-c33a-45df-9ce3-1a0c8763bb2e
@bind ix3 PlutoUI.Slider(1:size(lens_smooth, 2), default=size(lens_smooth, 1)÷2 + 1)

# ╔═╡ 7c968e03-5695-464b-aa8a-16e07c9c37b0
heatmap(Array(z1[:]) * 1e6, Array(x1[:]) * 1e6, lens_smooth[ix3, :, :], title="Smooth Lens - x slice at $(round(x1[ix3] * 1e6, digits=2))μm", xlabel="z in μm", ylabel="x in μm")

# ╔═╡ d03c9dbc-ca0a-467e-8efa-cd9dbbde07b9
md"## 5. Multi Slice with Smooth Boundary Lens"

# ╔═╡ c8a79512-a89d-4d0a-870f-51d28b794c86


# ╔═╡ b90bfc22-8e14-4220-8829-fc22abe45bfe


# ╔═╡ fea61f95-2fdb-49b0-b0ab-8160d5de3ef9


# ╔═╡ 8b99a4f8-7747-46a0-839f-76e83109e5dd
md"""## 6. Systematic Error of the Multi Slice Approach
This seems to be a systematic error of the Multi Slice approach?
Even for much smaller $z$ discretizations, it goes wrong.

As the refractive occurs mostly as a product with $\Delta z$, we would have expected a very small $\Delta z$ compensate for larger $\Delta n$. But clearly not.


Let's test the propagtion of a Gaussian beam in two media, refractive index of 1 and 1.5. If the multi slice approach would be correct, it should also handle the 1.5 medium correctly.

"""

# ╔═╡ 79c34154-2bbd-4056-9aa0-5574ec54e702
function multi_slice_1D(field, λ, L, z, medium; n=1)
	ΔL = L / length(field)
	fₓ = fftfreq(length(field), inv(ΔL))

	out_field = similar(field, (length(field), length(z) + 1))	

	out_field[:, 1] .= field
	Δz = z[2] - z[1]
	k = 2π / λ
	H = @. exp(1im * 2π * Δz  * √(Complex(n^2 / λ^2 - fₓ^2)))

	for i in 1:length(z)
		Δϕ = @. 1im * k * Δz * (medium[:, i] - n)
		out_field[:, i + 1] .= ifft(fft(out_field[:, i]) .* H)# .* exp.(Δϕ)
	end
	
	return out_field
end

# ╔═╡ 4c237907-c679-4ae2-b570-d4de519aaebf
gauss_1D = Complex.(exp.(- x_box.^2 / (2e-6)^2))

# ╔═╡ 1f9aaa64-a0b1-46e1-b8b3-cd57f7ef8002
medium_15 = ones((512, 200)) * 1.5;

# ╔═╡ 4fc00fd3-7334-4d91-9790-1c838a5d8254
medium_10 = ones((512, 200)) * 1.0;

# ╔═╡ 6c82815c-4059-49be-bf78-a08a6cdebe36
z_1D = range(0, 100e-6, 200);

# ╔═╡ 5e63ba45-9416-4ac0-afe9-9724bd1d84b0
gauss_1D_15 = multi_slice_1D(gauss_1D, 633e-9, 40e-6, z_1D, medium_15);

# ╔═╡ 3b28f6d9-bcf3-407d-b708-e80147d6875b
gauss_1D_10 = multi_slice_1D(gauss_1D, 633e-9, 40e-6, z_1D, medium_10);

# ╔═╡ 6c3c59bf-83da-4c28-a41c-227fbf7982cd
begin
	p1 = heatmap(z_1D * 1e6, x_box * 1e6, abs2.(gauss_1D_15)[:, begin:end-1].^0.2, xlabel="z in μm", ylabel="x in μm", title="n=1.0")
	p2 = heatmap(z_1D * 1e6, x_box * 1e6, abs2.(gauss_1D_10)[:, begin:end-1].^0.2, xlabel="z in μm", ylabel="x in μm", title="n=1.5")
	
	plot(p1, p2, layout=(1,2))
end

# ╔═╡ 8b594c1d-3404-44e5-9928-2e70912b9b8b
gauss_1D_15_2 = multi_slice_1D(gauss_1D, 633e-9, 40e-6, z_1D, medium_15; n=1.5);

# ╔═╡ 32007718-eaba-4bb9-8933-7ed836832a23
md"
The following Gedankenexperiment.


Let's assume we propagate a Gaussian beam in a medium with refractive index $n=1.5$.
So we slice the homogenous medium in thin slices with a $\Delta n = 0.5$.
In each step we propapate with the angular spectrum to the next slice. At each slice we multiply the phase by $\exp(i \cdot k \cdot \Delta z \cdot \Delta n)$. However, this will just do a global phase shift (a constant number), hence we can omit this skip.


What we obtain at the end, is just a beam propagated in vacuum and not in $n=1.5$ since at each multi-slice step the phase factor was constant.
Of course this solution is wrong, since the correct propagation kernel would have been the one with $k=\frac{2\pi}{\lambda} \cdot 1.5$ and not $k=\frac{2\pi}{\lambda}$. So independent of the size of $\Delta z$, the multi-slice approach will fail.
"

# ╔═╡ 984a6719-c67d-494e-b6ca-537d81decd7c
md"
### 1. Solution

To get the right propagated Gaussian beam, we need to use the right diffraction kernel with $n=1.5$

If we do so, the correct result will be obtained.
"

# ╔═╡ 13114099-4e02-496f-9f25-9a136ecd107a
begin
	heatmap(z_1D * 1e6, x_box * 1e6, abs2.(gauss_1D_15_2)[:, begin:end-1].^1, xlabel="z in μm", ylabel="x in μm", title="Gaussian Beam correct")
	plot!(z_1D * 1e6, @. 1e6 * 2e-6 * sqrt(1 + (z_1D / (π * 2e-6^2 * 1.5 / 633e-9))^2))
	plot!(z_1D * 1e6, @. -1e6 * 2e-6 * sqrt(1 + (z_1D / (π * 2e-6^2 * 1.5 / 633e-9))^2))
end

# ╔═╡ 99def499-4fcb-47a1-8c0c-3f26400707ba


# ╔═╡ f44cc074-3f93-4f0a-a584-6788215c5754
md"""# 4. Wave Propagation Method

!!! warn "Reference"
	Schmidt, S., et al. "Wave-optical modeling beyond the thin-element-approximation." Optics Express 24.26 (2016): 30188-30200.

Other researchers noticed this and published the wave propagation method [5] in 2016.
The mathematical issue of the spatially varying refractive index is, that the diffraction integral cannot be written as a Fourier transform:



$$E(x,y,z+\Delta z) = \frac{1}{2\pi} \int \widetilde{E}(k_x, k_y, z) \, e^{i k_z(k_x, k_y, x, y) \Delta z} \, e^{i (k_x x + k_y y)} \, \mathrm{d}k_x \, \mathrm{d}k_y$$

$$k_z(k_x, k_y, x, y) = \sqrt{k_0^2 n^2 \left( x,y,z + \frac{\Delta z}{2} \right) - k_x^2 - k_y^2},$$


Most optical systems (such as a thick lens) can be split into homogenous regions (such as air and glass).
The key idea, use in the respective region of the medium the correct refractive index! 

So for a medium with air and glass, we do the simulation two times. And then we stitch the field together.

Mathematically, with an index function the following can be achieved:

$$
I_m^z(x,y) = 
\begin{cases} 
1 & n_z(x,y) = n_m, \\
0 & n_z(x,y) \ne n_m,
\end{cases}
$$

$$E(x,y,z+\Delta z) = \sum_m I_m^z(x,y) \mathcal{F}^{-1} \left\{ e^{i k_z^m(k_x,k_y)\Delta z} \mathcal{F} \left\{ E(x,y,z) \right\} \right\},$$

$$k_z^m(k_x, k_y) = \sqrt{k_0^2 n_m^2 - k_x^2 - k_y^2} + \kappa(k_x, k_y).$$


"""

# ╔═╡ 24b316a9-fb50-4247-b160-03e6c14e830b
md" ## Results
In this case, the correct focal length (white line) is reproduced correctly.

"

# ╔═╡ da8af402-f368-41d5-99ad-7ccddd6b035e


# ╔═╡ d5c38a9c-9f98-4888-a5e5-1b61c99a17a3
md"""

# 5. Hankel Transform
!!! warn "Reference"
	Manuel Guizar-Sicairos and Julio C. Gutiérrez-Vega, \"Computation of quasi-discrete Hankel transforms of integer order for propagating optical wave fields,\" J. Opt. Soc. Am. A 21, 53-58 (2004) 

Let's recall that the Fourier transform can be written as

$$\mathcal{F}[f](k_x, k_y) = \int_{-\infty}^{\infty} F(x,y) \exp(i (k_x \cdot x + k_y \cdot y)) \,\mathrm{d}x\, \mathrm{d}y = \int_{-\infty}^{\infty} F(x,y) \exp(i \vec k \cdot \vec r) \,\mathrm{d}x\, \mathrm{d}y$$

If we transform this do polar coordinates $(r, \theta)$ and $(\kappa, \phi)$ we obtain

$$\mathcal{F}[f](\kappa, \phi) = \int_{0}^{\infty} \int_{0}^{2\pi} r f(r, \theta)\cdot \exp(i \cdot \cos(\theta - \phi) \kappa r)  \,\mathrm{d}\theta\, \mathrm{d}r$$

where $\kappa = \sqrt{k_x^2 + k_y^2}$ and $r=\sqrt{x^2 + y^2}$

We can now use

$$\exp(i x \cdot \sin(\theta)) = \sum_{n=-\infty}^{\infty} J_n(x) \cdot \exp(i n \theta)$$
where $J_n$ is the nth-order Bessel function of the first kind.

After integration (if $f$ has no $\theta$ dependency) over $\theta$ it only remains 

$$\mathcal{F}[f](\kappa, \phi) = 2\pi \int_{0}^{\infty} r \cdot f(r) \cdot J_0(\kappa \cdot r) \, \mathrm{d}r = \mathcal{H}[f](\rho)$$

where $\mathcal{H}$ is the Hankel transform.

This allows us to calculate the Fourier transform for the 2D case with the Fast Hankel transform in 1D. After the propagation, we can remap the cylindrical coordinates back to cartesian coordinates.

"""

# ╔═╡ 3df428e4-6b40-4602-a1fe-ac0ef7ea3401
md"## Implementation


"

# ╔═╡ 7475d1bc-4a84-4d8a-99ff-67287ca4760b
function qdht(f, _p, R::T, N) where T
	p = T(_p)
	α(p, k) = FunctionZeros.besselj_zero(p, k)
	ᾱ = [α(p, i) for i in 1:N]
	α_N_plus_1 = α(p, N+1) 
	V = α_N_plus_1/ (T(2π) * R)
	
	r̄ = ᾱ ./ (T(2π) * V)
	ν̄ = ᾱ ./ (T(2π) * R)
	S = α(p, N + 1)

	Tm = zeros(T, (N, N))
	for m in 1:N
		for n in m:N
			Tm[m, n] = 2 * besselj(p, ᾱ[m] * ᾱ[n] / S) / (abs(besselj(p + 1, ᾱ[n])) * abs(besselj(p + 1, ᾱ[m])) * S)
			Tm[n, m] = Tm[m, n]
		end
	end

	J̄ = abs.(besselj.(p + 1, ᾱ)) ./ R
	jp1 = abs.(besselj.(p + 1, ᾱ))
	jr = jp1 ./ R
	jv = jp1 ./ V

	function fwd(f)
		array = f.(r̄)
		a = jv .* (Tm * (array ./ jr))
		return a
	end

	function fwd_arr(array)
		a = jv .* (Tm * (array ./ jr))
		return a
	end

	function bwd(arr)
		b = jr .* (Tm * (arr ./ jv))
	end
	
	return ν̄, r̄, fwd, fwd_arr, bwd
end

# ╔═╡ 5e6ed976-1a80-4a85-bfc9-20d7936db26a
function radial_prop(f_input, z, λ::T, L, f_refractive_index; N=256, n0=T(1)) where T
	ν̄, r̄, fwd_f, fwd, bwd = qdht(f_input, 0, L, N)

	Δz = abs(z[2] - z[1])
	_f(ν) = exp(1im * T(2π) * Δz * sqrt(01im .+ 1 / λ^2 - ν^2))
	prop = _f.(ν̄)


	output = zeros(Complex{T}, (N, length(z) + 1))
	output[:, 1] = f_input.(r̄)

	dz = abs(z[2] - z[1])
	for i in 2:length(z) + 1
		field = output[:, i - 1]
		Δϕ = exp.(1im .* T(2π) ./ λ .* (f_refractive_index.(r̄, z[i - 1]; n0) .- n0) .* dz)
		output[:, i] .= bwd(prop .* fwd(field .* Δϕ))
	end

	return output, r̄
end

# ╔═╡ 3833ba9e-f6a1-4e60-bfa7-6e52e83fb5da
function radial_prop_WPM(f_input, z, λ::T, L, f_refractive_index; N=256, n0=T(1), n1=T(1.5)) where T
	ν̄, r̄, fwd_f, fwd, bwd = qdht(f_input, 0, L, N)

	Δz = abs(z[2] - z[1])
	_f(ν, n) = exp(1im * T(2π) * Δz * sqrt(0im .+ n^2 / λ^2 - ν^2))
	prop_1 = _f.(ν̄, n0)
	prop_2 = _f.(ν̄, n1)


	output = zeros(Complex{T}, (N, length(z) + 1))
	output[:, 1] = f_input.(r̄)

	dz = abs(z[2] - z[1])
	for i in 2:length(z) + 1
		field = output[:, i - 1]
		n = f_refractive_index.(r̄, z[i - 1]; n0)
		Δϕ1 = exp.(1im .* T(2π) ./ λ .* (n .- n0) .* dz)
		Δϕ2 = exp.(1im .* T(2π) ./ λ .* (n .- n1) .* dz)
		output1 = bwd(prop_1 .* fwd(field .* Δϕ1))
		output2 = bwd(prop_2 .* fwd(field .* Δϕ2))

		output[:, i] .= (n .≈ n0) .* output1 .+ (n .≈ n1) .* output2
	end

	return output, r̄
end

# ╔═╡ c165b464-b470-4e03-bdb1-05b70a24e9b9
function gaussian_beam(r::T, λ::T, w0::T, z) where T
    # Constants
    
    # Derived Parameters
    z_R = T(π) * w0^2 / λ  # Rayleigh range
    k = T(2π) / λ  # Wavenumber

    # Beam radius (w(z))
    w = w0 * sqrt(1 + (T(z) / z_R)^2)
    
    # Radius of curvature (R(z))
    R = z == 0 ? T(Inf) : z * (1 + (z_R / T(z))^2)
    
    # Gouy phase (ξ(z))
    ψ = atan(T(z) / z_R)
    
    # Intensity profile at given r and z
    intensity = (w0 / w) * exp(-r^2 / w^2) * exp(-1im *(k * T(z) + k * r^2 / 2 / R - ψ))

    return intensity
end

# ╔═╡ e8e428cc-2f7b-4bda-8a03-845631f55c43
function lens_function(r::T, z; n0=1, n_glass=T(1.5), radius=T(15e-6), offset=T(radius)) where T
	return ((r^2 + (z + offset)^2) <= radius^2) * (n_glass - n0) + n0
end

# ╔═╡ 9a6c1794-9963-4c85-bcf0-77b16f98a52a
z_gauss = range(-35f-6, 10f-6, 1024);

# ╔═╡ b3e0efeb-1a69-4249-813c-0abaf87e94c5
@time gb_propagated, r_gauss = radial_prop(x -> gaussian_beam(x, 633f-9, 15f-6, 0), z_gauss[:], 633f-9, 30f-6, lens_function; N=2048)

# ╔═╡ 83cf4959-1f04-4e5c-b84f-f035a2fac8fe
@time gb_propagated_wpm, r_gauss_wpm = radial_prop_WPM(x -> gaussian_beam(x, 633f-9, 15f-6, 0), z_gauss[:], 633f-9, 30f-6, lens_function; N=2048, n1=1.5f0)

# ╔═╡ b7793022-ee9b-4883-9870-c8aef2e82e49
begin
	p5 = heatmap(z_gauss[:], [.- reverse(r_gauss); r_gauss], abs2.([reverse(gb_propagated[:, 2:end], dims=1); gb_propagated[:, 2:end]]).^0.2)
	vline!(7.5f-6:7.5f-6, c=:white, title="Hankel Transform Multi Slice")
end

# ╔═╡ 75b73720-77f0-46eb-bf4e-b9e8fbe98d64
begin
		heatmap(z_gauss[:], [.- reverse(r_gauss); r_gauss], abs2.([reverse(gb_propagated_wpm[:, 2:end], dims=1); gb_propagated_wpm[:, 2:end]]).^0.2, title="Hankel Transform WPM")
		vline!(7.5f-6:7.5f-6, c=:white)
	
end

# ╔═╡ a11383a2-2483-4fb6-9cf5-bd3fe06f5d1a
md"""
# 6. Computational Complexities

### Multi Slice
If we have a field with $N \times N$ pixels, then the Angular Spectrum evalulates at a cost of two FFT transforms, which corresponds to $N^2 \cdot \log N$. But since we propagate $N_z$ steps, the total cost is $N_z \cdot N^2 \cdot \log N$.
The total memory cost is $N^2 \cdot N_z$

### Hankel Transform
The Hankel transform reduces this 2D field to a 1D field with only size $N_r$.
To propagate to another plane, we need two fast Hankel transform which evaluate each a matrix vector product. The total price for this is $N_r^2$. Since we do $N_z$ steps, we obtain $N_r^2 \cdot N_z$. The total memory cost is only $N_r \cdot N_z$.

"""

# ╔═╡ f609e529-d872-4bc0-b512-191ff3b3f4b9
md"""
## References
- J. M. Cowley and A. F. Moodie, “The Scattering of Electrons by Atoms and Crystals. I. A New Theoretical Approach,” Acta Cryst. 10 (1957)
- K. Li, M. Wojcik, and C. Jacobsen, “Multislice does it all—calculating the performance of nanofocusing X-ray optics,” Opt. Express 25, 1831 (2017)}
* Timothy M. Pritchett, Fast Hankel Transform Algorithms for Optical Beam Propagation, December 2001

"""

# ╔═╡ cef65203-91f6-4879-98b7-3419d1d98e12


# ╔═╡ 9f5993f1-b0e1-4188-bdfc-dd2ee1acb683


# ╔═╡ 428bb0ad-130c-41eb-a803-6f39b5003f78


# ╔═╡ 56a972e3-a5a7-40b3-941e-55f9124e6df0


# ╔═╡ 78c50adc-c698-4f12-a6f8-203787491e2f


# ╔═╡ 00cf9018-e9e3-42a4-b264-190e5e1abe99


# ╔═╡ 1961dc92-6b90-4f5f-a3b6-f5862fa00478
md"# Utility functions"

# ╔═╡ 3b82958b-9514-483d-b4a9-61a1cf0d0800
"""
    get_indices_around_center(i_in, i_out)

A function which provides two output indices `i1` and `i2`
where `i2 - i1 = i_out`

The indices are chosen in a way that the range `i1:i2`
cuts the interval `1:i_in` in a way that the center frequency
stays at the center position.

Works for both odd and even indices.
"""
function get_indices_around_center(i_in, i_out)
    if (mod(i_in, 2) == 0 && mod(i_out, 2) == 0 
     || mod(i_in, 2) == 1 && mod(i_out, 2) == 1) 
        x = (i_in - i_out) ÷ 2
        return 1 + x, i_in - x
    elseif mod(i_in, 2) == 1 && mod(i_out, 2) == 0
        x = (i_in - 1 - i_out) ÷ 2
        return 1 + x, i_in - x - 1 
    elseif mod(i_in, 2) == 0 && mod(i_out, 2) == 1
        x = (i_in - (i_out - 1)) ÷ 2
        return 1 + x, i_in - (x - 1)
    end
end

# ╔═╡ 58a545b3-a5f5-4ed6-8132-6f1384882aa6
"""
	    set_center!(arr_large, arr_small; broadcast=false)
	
	Puts the `arr_small` central into `arr_large`.
	
	The convention, where the center is, is the same as the definition
	as for FFT based centered.
	
	Function works both for even and uneven arrays.
	See also [`crop_center`](@ref), [`pad`](@ref), [`set_center!`](@ref).
	
	# Keyword
	* If `broadcast==false` then a lower dimensional `arr_small` will not be broadcasted
	along the higher dimensions.
	* If `broadcast==true` it will be broadcasted along higher dims.
	
	
	See also [`crop_center`](@ref), [`pad`](@ref).
	
	
	# Examples
	```jldoctest
	julia> set_center!([1, 1, 1, 1, 1, 1], [5, 5, 5])
	6-element Vector{Int64}:
	 1
	 1
	 5
	 5
	 5
	 1
	
	julia> set_center!([1, 1, 1, 1, 1, 1], [5, 5, 5, 5])
	6-element Vector{Int64}:
	 1
	 5
	 5
	 5
	 5
	 1
	
	julia> set_center!(ones((3,3)), [5])
	3×3 Matrix{Float64}:
	 1.0  1.0  1.0
	 1.0  5.0  1.0
	 1.0  1.0  1.0
	
	julia> set_center!(ones((3,3)), [5], broadcast=true)
	3×3 Matrix{Float64}:
	 1.0  1.0  1.0
	 5.0  5.0  5.0
	 1.0  1.0  1.0
	```
	"""
	function set_center!(arr_large::AbstractArray{T, N}, arr_small::AbstractArray{T1, M};
	                     broadcast=false) where {T, T1, M, N}
	    @assert N ≥ M "Can't put a higher dimensional array in a lower dimensional one."
	
	    if broadcast == false
	        inds = ntuple(i -> begin
	                        a, b = get_indices_around_center(size(arr_large, i), size(arr_small, i))
	                        a:b
	                      end,
	                      Val(N)) 
	        arr_large[inds..., ..] .= arr_small
	    else
	        inds = ntuple(i -> begin
	                        a, b = get_indices_around_center(size(arr_large, i), size(arr_small, i))
	                        a:b
	                      end,
	                      Val(M)) 
	        arr_large[inds..., ..] .= arr_small
	    end
	
	    
	    return arr_large
	end

# ╔═╡ f101181f-e10b-41a6-bb31-2476ac2257af
"""
	    crop_center(arr, new_size)
	
	Extracts a center of an array. 
	`new_size_array` must be list of sizes indicating the output
	size of each dimension. Centered means that a center frequency
	stays at the center position. Works for even and uneven.
	If `length(new_size_array) < length(ndims(arr))` the remaining dimensions
	are untouched and copied.
	
	
	See also [`pad`](@ref), [`set_center!`](@ref).
	
	# Examples
	```jldoctest
	julia> crop_center([1 2; 3 4], (1,))
	1×2 Matrix{Int64}:
	 3  4
	
	julia> crop_center([1 2; 3 4], (1,1))
	1×1 Matrix{Int64}:
	 4
	
	julia> crop_center([1 2; 3 4], (2,2))
	2×2 Matrix{Int64}:
	 1  2
	 3  4
	```
	"""
	function crop_center(arr, new_size::NTuple{N}; return_view=true) where {N}
	    M = ndims(arr)
	    @assert N ≤ M "Can't specify more dimensions than the array has."
	    @assert all(new_size .≤ size(arr)[1:N]) "You can't extract a larger array than the input array."
	    @assert Base.require_one_based_indexing(arr) "Require one based indexing arrays"
	
	    out_indices = ntuple(i ->   let  
	                                    if i ≤ N
	                                        inds = get_indices_around_center(size(arr, i),
	                                                                         new_size[i])
	                                        inds[1]:inds[2]
	                                    else
	                                        1:size(arr, i)
	                                    end
	                                end,
	                          Val(M))
	    
	
	    # if return_view
	        # return @inbounds view(arr, out_indices...)
	    # else
	        return @inbounds arr[out_indices...]
	    # end
	end

# ╔═╡ 17a0fb88-f01f-4940-85bb-1283f0bfc74e
"""
	plan_multi_slice(beam::AbstractArray{CT, 2}, medium, z, λ, L; n0=1)

Return an efficient function to propagate an initial `beam` through a `medium` with refractive index distribution. We do the multi slice angular spectrum approach.

`z` is the `AbstractVector` containing the z distances. `λ` is the vacuum wavelength and `L` is the total x,y size of the beam.

`n0` is the average refractive index used for propagation.
"""
function plan_multi_slice(beam::AbstractArray{CT, 2}, medium, z, λ, L; n0=1) where CT

	buffer1 = similar(beam, 2 .* size(beam)[1:2])
	buffer2 = similar(buffer1)
	p = plan_fft(buffer1, (1,2))

	M = 2
	L_pad = M * L


	k = CT(2π) / λ * n0
	Ns = size(beam)[1:2] .* M
	# sample spacing
	dx = L_pad ./ Ns
	# frequency spacing
	df = 1 ./ L_pad
    # total size in frequency space
	Lf = Ns .* df
	# frequencies centered around first entry 
	# 1D vectors each
	f_y = similar(beam, real(eltype(beam)), (Ns[1], 1))
	f_y .= fftfreq(Ns[1], Lf[1])
	f_x = similar(beam, real(eltype(beam)), (1, Ns[2]))
	f_x .= fftfreq(Ns[2], Lf[2])'

	CUDA.@allowscalar dz = abs(z[2] - z[1])

	H = exp.(1im .* k .* abs.(dz) .* sqrt.(CT(1) .- abs2.(f_x .* λ ./ n0) .- abs2.(f_y .* λ ./ n0)))


	f = let p=p, H=H, z=z, buffer1=buffer1, buffer2=buffer2, dz=dz, k=k
		function f(field)
			field_history = similar(field, (size(field)[1:2]..., length(z) + 1))
			field_history[:, :, 1] .= field
			for i in 1:length(z)
				field = field_history[:, :, i]
				if !isnothing(medium)
					field .*= @views exp.(1im .* k .* (medium[:, :, i] .- n0) .* dz)
				end
			    fill!(buffer2, 0)
			    fieldp = set_center!(buffer2, field)
			    field_imd = p * ifftshift!(buffer1, fieldp, (1, 2))	
			    field_imd .*= H
			    field_out = fftshift!(buffer2, inv(p) * field_imd, (1, 2))
			    field_out_cropped = crop_center(field_out, size(field))
				field_history[:, :, i + 1] .= field_out_cropped
			end
			return field_history
		end
	end
	return f, H
end

# ╔═╡ 79d5496c-d9a1-442e-b69a-ffac33288ab1
MS, H = plan_multi_slice(beam, lens1, z1[:], λ, L1, n0=1.f0)

# ╔═╡ 71f0f949-32ec-4cb0-ae8a-5c1091d82796
@time result = MS(beam);

# ╔═╡ 611a6aba-1c2f-4c05-ae36-3c646fdaedc4
begin
	heatmap(Array(z1)[:] * 1e6, Array(x1)[:] * 1e6, Array(abs2.(result[:, size(result, 2)÷2 + 1, begin:end-1])).^0.2, grid=:white, xlabel="z in μm", ylabel="x in μm")
	vline!((7.5f-6:7.5f-6) * 1e6, c=:white, label="real focus")
	#heatmap(Array(z1)[begin:end-1], Array(x1)[:], Array(lens1[:, 257, begin:end-1]), grid=:white)
	plot!(sin.(0:0.01:2π) .* 15 .- 15 , cos.(0:0.01:2π) .* 15, label="lens")

end

# ╔═╡ 150cae90-4709-4adc-9933-d2bd073d49a2
MS_smooth, _ = plan_multi_slice(beam, (lens_smooth), z1[:], λ, L1, n0=1.0f0)

# ╔═╡ 2155a928-59e5-42b5-b98c-c83133260b18
@time result_smooth = MS_smooth(beam);

# ╔═╡ 7b5d3741-1f0f-42f1-9aba-24d5b3630a9b
begin
	heatmap(Array(z1)[:] * 1e6, Array(x1)[:] * 1e6, Array(abs2.(result_smooth[:, size(result_smooth, 2)÷2 + 1, begin:end-1])).^0.2, grid=:white)
	vline!(7.5:7.5, c=:white, xlabel="z in μm", ylabel="x in μm", label="real focus")
	#heatmap(Array(z1)[begin:end-1], Array(x1)[:], Array(lens1[:, 257, begin:end-1]), grid=:white)
	plot!(sin.(0:0.01:2π) .* 15 .- 15 , cos.(0:0.01:2π) .* 15, label="lens")
end

# ╔═╡ 6fd65d63-65c9-4bc9-8259-376387e0b69d
"""
	plan_multi_slice(beam::AbstractArray{CT, 2}, medium, z, λ, L; n0=1)

Return an efficient function to propagate an initial `beam` through a `medium` with refractive index distribution. We do the wave propagation method.

`z` is the `AbstractVector` containing the z distances. `λ` is the vacuum wavelength and `L` is the total x,y size of the beam.

`n0` is the average refractive index used for propagation.
"""
function plan_WPM(beam::AbstractArray{CT, 2}, medium, z, λ, L; n0=1, n_lens=1.5f0, medium_index=medium) where CT

	buffer1 = similar(beam, (2 .* size(beam)[1:2]..., 2))
	buffer2 = similar(buffer1)
	p = plan_fft(buffer1, (1,2))

	M = 2
	L_pad = M * L


	k = CT(2π) / λ

	Ns = size(beam)[1:2] .* M
	# sample spacing
	dx = L_pad ./ Ns
	# frequency spacing
	df = 1 ./ L_pad
    # total size in frequency space
	Lf = Ns .* df
	# frequencies centered around first entry 
	# 1D vectors each
	f_y = similar(beam, real(eltype(beam)), (Ns[1], 1))
	f_y .= fftfreq(Ns[1], Lf[1])
	f_x = similar(beam, real(eltype(beam)), (1, Ns[2]))
	f_x .= fftfreq(Ns[2], Lf[2])'

	CUDA.@allowscalar dz = abs(z[2] - z[1])

	
	H_air = exp.(1im .* k .* n0 .*abs.(dz) .* sqrt.(CT(1) .- abs2.(f_x .* λ ./ n0) .- abs2.(f_y .* λ ./ n0)))
	
	H_lens = exp.(1im .* k .* n_lens .* abs.(dz) .* sqrt.(CT(1) .- abs2.(f_x .* λ ./ n_lens) .- abs2.(f_y .* λ ./ n_lens)))

	H = cat(H_air, H_lens, dims=3)

	f = let p=p, H=H, z=z, buffer1=buffer1, buffer2=buffer2, dz=dz, k=k
		function f(_field)
			field = repeat(_field, 1,1,2)
			
			field_history = similar(_field, (size(_field)[1:2]..., size(medium, 3) + 1))
			field_history[:, :, 1] .= field[:, :, 1]
			for i in axes(medium, 3)
				field .= field_history[:, :, i]
				# apply small multi slice step
				field[:, :, 1] .*= @views exp.(1im .* k .* (medium[:, :, i] .- n0) .* dz)
				field[:, :, 2] .*= @views exp.(1im .* k .* (medium[:, :, i] .- n_lens) .* dz)

				# convolution here
			    fill!(buffer2, 0)
			    fieldp = set_center!(buffer2, field)
			    field_imd = p * ifftshift!(buffer1, fieldp, (1, 2))
			    field_imd .*= H
			    field_out = fftshift!(buffer2, inv(p) * field_imd, (1, 2))
				field_out_cropped = crop_center(field_out, (size(field)))

				# assemble field parts
				field_history[:, :, i + 1] .= field_out_cropped[:, :, 1] .* (medium_index[:, :, i] .≈ n0) .+  field_out_cropped[:, :, 2] .* (medium_index[:, :, i] .≈ n_lens)
			end
			return field_history
		end
	end
	return f, H
end

# ╔═╡ ec4a4ae1-b201-4e1c-bbc0-16f47813240e
WPM, _ = plan_WPM(beam, lens1, z1[:], λ, L1, n0=1f0, n_lens=n_glass)

# ╔═╡ ff7bc968-4fb0-4e0c-9e99-25d9298a6d93
@time result_wpm = WPM(beam);

# ╔═╡ 1969105b-a579-4f05-a7b6-4af7cb1b259e
sum(result_wpm[:,:,1])

# ╔═╡ 98f00b68-07bd-4861-a820-8de833d15444
begin
	heatmap(Array(z1)[:] * 1e6, Array(x1)[:] * 1e6, Array(abs2.(result_wpm[:, size(result_wpm, 2)÷2 +1, begin:end-1])).^0.2, grid=:white, xlabel="z in μm", ylabel="x in μm", title="WPM")
	vline!((7.5f-6:7.5f-6) * 1e6, c=:white)
	#heatmap(Array(lens1[:, size(result_wpm, 2)÷2 +1, begin:end-1]), grid=:white)
end

# ╔═╡ f32830eb-e0e1-4a9a-8a73-520e7a9f8bdd
WPM_smooth, _ = plan_WPM(beam, lens_smooth, z1[:], λ, L1, n0=1f0, n_lens=n_glass, medium_index=lens1)

# ╔═╡ df7c0a6b-5308-41bf-8f9b-e308aa7169c1
@time result_wpm_smooth = WPM_smooth(beam);

# ╔═╡ 9615bb66-844e-4ff6-a299-4b558a9fdc01
begin
	heatmap(Array(z1)[:] * 1e6, Array(x1)[:] * 1e6, Array(abs2.(result_wpm_smooth[:, size(result_wpm, 2)÷2 +1, begin:end-1])).^0.2, grid=:white, xlabel="z in μm", ylabel="x in μm", title="WPM Smooth Lens")
	vline!((7.5f-6:7.5f-6) * 1e6, c=:white)
	#heatmap(Array(lens1[:, size(result_wpm, 2)÷2 +1, begin:end-1]), grid=:white)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
EllipsisNotation = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
FunctionZeros = "b21f74c0-b399-568f-9643-d20f4fa2c814"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
IndexFunArrays = "613c443e-d742-454e-bfc6-1d7f8dd76566"
NDTools = "98581153-e998-4eef-8d0d-5ec2c052313d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
CUDA = "~5.5.2"
EllipsisNotation = "~1.8.0"
FFTW = "~1.8.0"
FileIO = "~1.16.6"
FunctionZeros = "~1.0.0"
ImageIO = "~0.6.9"
ImageShow = "~0.3.8"
IndexFunArrays = "~0.2.7"
NDTools = "~0.7.0"
Plots = "~1.40.8"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.60"
SpecialFunctions = "~2.4.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "4a1ac22c9a98c10d94ee494a08df2cd25d5d8601"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "b392ede862e506d451fc1616e79aa6f4c673dab8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.38"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.Aqua]]
deps = ["Compat", "Pkg", "Test"]
git-tree-sha1 = "49b1d7a9870c87ba13dc63f8ccfcf578cb266f95"
uuid = "4c88cf16-eb10-579e-8560-4a9242c79595"
version = "0.8.9"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "d60a1922358aa203019b7857a2c8c37329b8736c"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.17.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "492681bc44fac86804706ddb37da10880a2bd528"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.10.4"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "2c7cc21e8678eff479978a0a2ef5ce2f51b63dff"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "d434647f798823bcae510aee0bc0401927f64391"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.1.1"

    [deps.BlockArrays.extensions]
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "e0725a467822697171af4dae15cec10b4fc19053"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.5.2"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ccd1e54610c222fadfd4737dac66bff786f63656"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.10.3+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "33576c7c1b2500f8e7e6baa082e04563203b3a45"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.3.5"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "e43727b237b2879a34391eeb81887699a26f8f2f"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.15.3+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EllipsisNotation]]
deps = ["StaticArrayInterface"]
git-tree-sha1 = "3507300d4343e8e4ad080ad24e335274c2e297a9"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.8.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.FunctionZeros]]
deps = ["Roots", "SpecialFunctions"]
git-tree-sha1 = "0acddff2143204e318186edda996fa8615e1cabc"
uuid = "b21f74c0-b399-568f-9643-d20f4fa2c814"
version = "1.0.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "62ee71528cca49be797076a76bdc654a170a523e"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "10.3.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "1d6f290a5eb1201cd63574fbc4440c788d5cb38f"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.27.8"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee28ddcd5517d54e417182fec3886e7412d3926f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f31929b9e67066bee48eec8b03c0df47d31a74b3"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0224cce99284d997f6880a42ef715a37c99338d1"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.2+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "1336e07ba2eb75614c99496501a8f4b233e9fafe"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.10"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["Aqua", "BlockArrays", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "661ca04f8df633e8a021c55a22e96cf820220ede"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndexFunArrays]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f78703c7a4ba06299cddd8694799c91de0157ac"
uuid = "613c443e-d742-454e-bfc6-1d7f8dd76566"
version = "0.2.7"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "39d64b09147620f5ffbf6b2d3255be3c901bec63"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.8"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "10da5154188682e5c0726823c2b5125957ec3778"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.38"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "e73a077abc7fe798fe940deabe30ef6c66bdde52"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.29"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d422dfd9707bec6617335dc2ea3c5172a87d5908"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.3"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6ce1e19f3aec9b59186bdf06cdf3c4fc5f5f3e6"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.50.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NDTools]]
deps = ["LinearAlgebra", "OffsetArrays", "PaddedViews", "Random", "Statistics"]
git-tree-sha1 = "6ec3344ccc0d76354824ccfce80d3568e1a80138"
uuid = "98581153-e998-4eef-8d0d-5ec2c052313d"
version = "0.7.0"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "53046f0483375e3ed78e49190f1154fa0a4083a1"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "0.3.4"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ce3269ed42816bf18d500c9f63418d4b0d9f5a3b"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.1.0+2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "45470145863035bb124ca51b320ed35d071cc6c2"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.8"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "470f48c9c4ea2170fd4d0f8eb5118327aada22f5"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.4"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "3a7c7e5c3f015415637f5debdf8a674aa2c979c4"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.1"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "52af86e35dd1b177d051b12681e1c581f53c281b"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "d0553ce4031a081cc42387a9b9c8441b7d99f32d"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.7"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "0248b1b2210285652fbc67fd6ced9bf0394bcfec"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.1"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3a6f063d690135f5c1ba351412c82bae4d1402bf"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.25"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "2d17fabcd17e67d7625ce9c531fb9f40b7c42ce4"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.2.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "6a451c6f33a176150f315726eba8b92fbfdb9ae7"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.4+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "15e637a697345f6743674f1322beefbc5dcd5cfc"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "936081b536ae4aa65415d869287d43ef3cb576b2"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.53.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "7dfa0fd9c783d3d0cc43ea1af53d69ba45c447df"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─1bc50357-ec6a-404a-a2b9-b8d31b153475
# ╠═7112c7f2-a35a-11ef-0005-b77b37b5bb2c
# ╠═55fc243d-b162-43fa-b755-846ab413d530
# ╠═15fe405d-a554-44b4-a527-45ddcbcdfad8
# ╠═e0ad01cc-622d-4505-ba8b-29b504f92300
# ╠═526365f7-9a74-4c7c-9d9e-95fd7de1b549
# ╠═323c620a-0652-41e6-9190-8d548cba3a1c
# ╠═f4fa6dab-d64f-46be-9f13-6cccbf50400a
# ╠═00b13143-d9bd-402f-858c-0270a012a61f
# ╠═6ce9a70c-0162-480a-9177-2715ff2f963a
# ╟─57387f24-15a4-4ea8-867e-e7d524c310b4
# ╟─443840db-b308-4202-b47d-7a96779f6648
# ╟─d09a7268-e3f6-4ced-8cca-24b718895c19
# ╠═5c16ec60-8531-40a6-9dde-8997b13f91e2
# ╠═bcf50548-670b-43dd-8d10-0763aeac423b
# ╠═069ea1ca-d759-4fde-addb-ebf8cefd1596
# ╠═6dcc011d-6d91-401f-b4ef-f78ddce62398
# ╠═24414435-617b-4fa5-a500-850ec67da0dc
# ╟─377246ab-19ba-4636-b527-83deac269574
# ╟─2d4d4582-65cf-48fa-b369-a808627feb6a
# ╟─6d3b05ba-c1c9-475b-82c8-6ffd1175f161
# ╠═d259a222-bdd1-4af1-8a32-be942551dc77
# ╟─185ac123-226f-443c-9882-c758677c120a
# ╟─7b6a90ce-c32d-495d-b642-2f5baa7c3e05
# ╠═5bc6b3ae-29d5-4df7-8f35-2154ca0ef34b
# ╠═47d0d3d3-83f5-4263-af38-b21a20714744
# ╠═5503ca5c-114e-434b-ab69-37d50e09bfea
# ╠═ac6dcc8a-e940-490b-b697-c61bcde333ec
# ╟─6f7ad76b-a21c-4413-97e8-6a3b33c13a00
# ╠═00509bfa-80af-4a4c-b246-a211d0ff1ca1
# ╠═9bdb1c18-12d6-45c4-8048-597c1926e056
# ╠═3f76ed4a-2c67-4d16-b6ba-a3388dd3a0c4
# ╟─17a0fb88-f01f-4940-85bb-1283f0bfc74e
# ╟─622af1b1-be09-404d-a566-0954c0ca486f
# ╟─209d08ae-168e-48e0-a4b7-6f842c291a73
# ╟─8e798c1d-9386-4446-bb12-11d0a7de0277
# ╠═1397b1f7-599b-4df5-ad4d-8f626e527a26
# ╠═97f4e89f-b946-41a8-bfa8-b5325580e143
# ╠═bdae79e7-5976-4a2e-b3da-be809d8de95f
# ╠═fe605fa6-b394-4531-8fff-93d7e755e8bb
# ╠═ca49b92f-b376-47a0-ab7b-b390eaa15bdf
# ╠═b796d8c5-dba2-4b5a-9914-80566a5e004d
# ╠═28e3d7bc-df79-4618-b723-9766706e1726
# ╠═ed9fce6a-416c-41a5-a3df-cbd380dee960
# ╠═94fe19f5-dea0-48aa-9b64-e38961ea1176
# ╟─c22b7856-ec84-44b6-91ed-3fcdf7653478
# ╟─0f7e509f-804a-45f2-8149-792bb2583d16
# ╟─4a117563-1ec8-4daf-b298-9e648e0e2990
# ╠═589bf363-91b6-4381-902b-092da1e8c96e
# ╠═ddf01f2c-8b7e-459c-86c5-f01f1c6aaa3c
# ╠═36537308-0263-4555-9096-a90786849554
# ╟─0537bafa-e13d-4ff1-92bc-6bbc35baa5d2
# ╟─dc9b64ab-f619-4837-a6de-6fdd345a4d89
# ╠═79d5496c-d9a1-442e-b69a-ffac33288ab1
# ╠═71f0f949-32ec-4cb0-ae8a-5c1091d82796
# ╟─611a6aba-1c2f-4c05-ae36-3c646fdaedc4
# ╟─ba492d7e-bf99-43b5-88e2-b6c23b24c89d
# ╟─eb06dc36-0f3a-4841-b7e9-31bd8f904cf7
# ╠═97b68214-e2dc-4805-87a7-44526ebe5715
# ╟─d6125cbc-1def-4b8d-b4e1-1275ea304843
# ╟─3c8b36b0-c777-4b03-b2d6-0cfa19e6b40c
# ╟─25c26f57-c5a4-4137-a8e2-407cbc9c7dbe
# ╠═c2aac6a8-8daa-4cac-adff-e7dec47992b2
# ╟─f4d61401-c33a-45df-9ce3-1a0c8763bb2e
# ╟─7c968e03-5695-464b-aa8a-16e07c9c37b0
# ╟─d03c9dbc-ca0a-467e-8efa-cd9dbbde07b9
# ╠═150cae90-4709-4adc-9933-d2bd073d49a2
# ╠═2155a928-59e5-42b5-b98c-c83133260b18
# ╟─7b5d3741-1f0f-42f1-9aba-24d5b3630a9b
# ╠═c8a79512-a89d-4d0a-870f-51d28b794c86
# ╠═b90bfc22-8e14-4220-8829-fc22abe45bfe
# ╠═fea61f95-2fdb-49b0-b0ab-8160d5de3ef9
# ╟─8b99a4f8-7747-46a0-839f-76e83109e5dd
# ╠═79c34154-2bbd-4056-9aa0-5574ec54e702
# ╠═4c237907-c679-4ae2-b570-d4de519aaebf
# ╠═1f9aaa64-a0b1-46e1-b8b3-cd57f7ef8002
# ╠═4fc00fd3-7334-4d91-9790-1c838a5d8254
# ╠═6c82815c-4059-49be-bf78-a08a6cdebe36
# ╠═5e63ba45-9416-4ac0-afe9-9724bd1d84b0
# ╠═3b28f6d9-bcf3-407d-b708-e80147d6875b
# ╟─6c3c59bf-83da-4c28-a41c-227fbf7982cd
# ╠═8b594c1d-3404-44e5-9928-2e70912b9b8b
# ╟─32007718-eaba-4bb9-8933-7ed836832a23
# ╟─984a6719-c67d-494e-b6ca-537d81decd7c
# ╟─13114099-4e02-496f-9f25-9a136ecd107a
# ╠═99def499-4fcb-47a1-8c0c-3f26400707ba
# ╟─f44cc074-3f93-4f0a-a584-6788215c5754
# ╟─6fd65d63-65c9-4bc9-8259-376387e0b69d
# ╠═ec4a4ae1-b201-4e1c-bbc0-16f47813240e
# ╠═f32830eb-e0e1-4a9a-8a73-520e7a9f8bdd
# ╠═ff7bc968-4fb0-4e0c-9e99-25d9298a6d93
# ╠═df7c0a6b-5308-41bf-8f9b-e308aa7169c1
# ╠═1969105b-a579-4f05-a7b6-4af7cb1b259e
# ╟─24b316a9-fb50-4247-b160-03e6c14e830b
# ╟─98f00b68-07bd-4861-a820-8de833d15444
# ╟─9615bb66-844e-4ff6-a299-4b558a9fdc01
# ╠═da8af402-f368-41d5-99ad-7ccddd6b035e
# ╟─d5c38a9c-9f98-4888-a5e5-1b61c99a17a3
# ╟─3df428e4-6b40-4602-a1fe-ac0ef7ea3401
# ╠═7475d1bc-4a84-4d8a-99ff-67287ca4760b
# ╠═5e6ed976-1a80-4a85-bfc9-20d7936db26a
# ╠═3833ba9e-f6a1-4e60-bfa7-6e52e83fb5da
# ╠═c165b464-b470-4e03-bdb1-05b70a24e9b9
# ╠═e8e428cc-2f7b-4bda-8a03-845631f55c43
# ╠═9a6c1794-9963-4c85-bcf0-77b16f98a52a
# ╠═b3e0efeb-1a69-4249-813c-0abaf87e94c5
# ╠═83cf4959-1f04-4e5c-b84f-f035a2fac8fe
# ╟─b7793022-ee9b-4883-9870-c8aef2e82e49
# ╟─75b73720-77f0-46eb-bf4e-b9e8fbe98d64
# ╟─a11383a2-2483-4fb6-9cf5-bd3fe06f5d1a
# ╠═f609e529-d872-4bc0-b512-191ff3b3f4b9
# ╠═cef65203-91f6-4879-98b7-3419d1d98e12
# ╠═9f5993f1-b0e1-4188-bdfc-dd2ee1acb683
# ╠═428bb0ad-130c-41eb-a803-6f39b5003f78
# ╠═56a972e3-a5a7-40b3-941e-55f9124e6df0
# ╠═78c50adc-c698-4f12-a6f8-203787491e2f
# ╠═00cf9018-e9e3-42a4-b264-190e5e1abe99
# ╟─1961dc92-6b90-4f5f-a3b6-f5862fa00478
# ╟─58a545b3-a5f5-4ed6-8132-6f1384882aa6
# ╟─f101181f-e10b-41a6-bb31-2476ac2257af
# ╠═3b82958b-9514-483d-b4a9-61a1cf0d0800
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
