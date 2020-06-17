module ParXRec

using ..LightXML

export parxrec, parrec, xmlrec, SeriesInfo, ImageInfo, RecHdr

Base.@kwdef mutable struct SeriesInfo
    Patient_Name               :: String  = ""
    Examination_Name           :: String  = ""
    Protocol_Name              :: String  = ""
    Examination_Date           :: String  = ""      # TODO: Date
    Examination_Time           :: String  = ""      # TODO: Time
    Series_Data_Type           :: String  = ""
    Aquisition_Number          :: Int32   = 0
    Reconstruction_Number      :: Int32   = 0
    Scan_Duration              :: Float32 = 0
    Max_No_Phases              :: Int32   = 1
    Max_No_Echoes              :: Int32   = 1
    Max_No_Slices              :: Int32   = 1
    Max_No_Dynamics            :: Int32   = 1
    Max_No_Mixes               :: Int16   = 1
    Max_No_B_Values            :: Int32   = 1
    Max_No_Gradient_Orients    :: Int32   = 1
    No_Label_Types             :: Int32   = 0
    Patient_Position           :: String  = ""
    Preparation_Direction      :: String  = ""
    Technique                  :: String  = ""
    Scan_Resolution_X          :: Int16   = 0
    Scan_Resolution_Y          :: Int32   = 0
    Scan_Mode                  :: String  = ""
    Repetition_Times           :: NTuple{2,Float32} = (0,0)
    FOV_AP                     :: Float32 = 0
    FOV_FH                     :: Float32 = 0
    FOV_RL                     :: Float32 = 0
    Water_Fat_Shift            :: Float32 = 0
    Angulation_AP              :: Float32 = 0
    Angulation_FH              :: Float32 = 0
    Angulation_RL              :: Float32 = 0
    Off_Center_AP              :: Float32 = 0
    Off_Center_FH              :: Float32 = 0
    Off_Center_RL              :: Float32 = 0
    Flow_Compensation          :: Bool    = false
    Presaturation              :: Bool    = false
    Phase_Encoding_Velocity    :: NTuple{3,Float32} = (0,0,0)
    MTC                        :: Bool    = false
    SPIR                       :: Bool    = false
    EPI_factor                 :: Int32   = 0
    Dynamic_Scan               :: Bool    = false
    Diffusion                  :: Bool    = false
    Diffusion_Echo_Time        :: Float32 = 0
    PhotometricInterpretation  :: String  = ""
end

Base.@kwdef mutable struct ImageKey
    Slice       :: Int32  = 1
    Echo        :: Int32  = 1
    Dynamic     :: Int32  = 1
    Phase       :: Int32  = 1
    BValue      :: Int32  = 1
    Grad_Orient :: Int32  = 1
    Label_Type  :: String = ""      # Enumeration
    Type        :: String = ""      # Enumeration
    Sequence    :: String = ""      # Enumeration
    Index       :: Int32  = 0
end

Base.@kwdef mutable struct ImageInfo
    Key                        :: ImageKey
    Pixel_Size                 :: UInt16  = 0
    Scan_Percentage            :: Float64 = 0
    Resolution_X               :: UInt16  = 0
    Resolution_Y               :: UInt16  = 0
    Rescale_Intercept          :: Float64 = 0
    Rescale_Slope              :: Float64 = 0
    Scale_Slope                :: Float32 = 0
    Window_Center              :: Float64 = 0
    Window_Width               :: Float64 = 0
    Slice_Thickness            :: Float64 = 0
    Slice_Gap                  :: Float64 = 0
    Display_Orientation        :: String  = ""      # Enumeration
    fMRI_Status_Indication     :: Int16   = 0
    Image_Type_Ed_Es           :: String  = ""      # Enumeration
    Pixel_Spacing              :: NTuple{2,Float64} = (0,0)
    Echo_Time                  :: Float64 = 0
    Dyn_Scan_Begin_Time        :: Float32 = 0
    Trigger_Time               :: Float64 = 0
    Diffusion_B_Factor         :: Float32 = 0
    No_Averages                :: Float64 = 0
    Image_Flip_Angle           :: Float64 = 0
    Cardiac_Frequency          :: Int32   = 0
    Min_RR_Interval            :: Int32   = 0
    Max_RR_Interval            :: Int32   = 0
    TURBO_Factor               :: Int32   = 0
    Inversion_Delay            :: Float64 = 0
    Contrast_Type              :: String  = ""
    Diffusion_Anisotropy_Type  :: String  = ""
    Diffusion_AP               :: Float32 = 0
    Diffusion_FH               :: Float32 = 0
    Diffusion_RL               :: Float32 = 0
    Angulation_AP              :: Float64 = 0
    Angulation_FH              :: Float64 = 0
    Angulation_RL              :: Float64 = 0
    Offcenter_AP               :: Float64 = 0
    Offcenter_FH               :: Float64 = 0
    Offcenter_RL               :: Float64 = 0
    Slice_Orientation          :: String  = ""      # Enumeration
    Image_Planar_Configuration :: UInt16  = 0
    Samples_Per_Pixel          :: UInt16  = 0
end

mutable struct RecHdr
    series :: SeriesInfo
    images :: Vector{ImageInfo}
end

####
#### ParXRec
####

parxrec(path::AbstractString; load_data::Bool = true) =
    parxrec(Float64, path, load_data = load_data)

function parxrec(T::DataType, path::AbstractString; load_data::Bool = true)
    hdrpath, recpath = parxrec_paths(path, load_data)
    ext = splitext(hdrpath)[2]
    hdr = lowercase(ext) == ".par" ? readpar(hdrpath) : readxml(hdrpath)
    return load_data ? (hdr, readrec(T, recpath, hdr)) : hdr
end

parrec(path::AbstractString; load_data::Bool = true) =
    parrec(Float64, path, load_data = load_data)

function parrec(T::DataType, path::AbstractString; load_data::Bool = true)
    hdrpath, recpath = parxrec_paths(path, load_data, ".PAR")
    hdr = readpar(hdrpath)
    return load_data ? (hdr, readrec(T, recpath, hdr)) : hdr
end

xmlrec(path::AbstractString; load_data::Bool = true) =
    xmlrec(Float64, path, load_data = load_data)

function xmlrec(T::DataType, path::AbstractString; load_data::Bool = true)
    hdrpath, recpath = parxrec_paths(path, load_data, ".XML")
    hdr = readxml(hdrpath)
    return load_data ? (hdr, readrec(T, recpath, hdr)) : hdr
end

function parxrec_paths(path::AbstractString, load_data::Bool, ext = "")
    isfile(path) || error("file $path does not exist")
    path1, ext1 = splitext(path)

    if lowercase(ext1) == ".xml" || lowercase(ext1) == ".par"
        ext == "" || lowercase(ext1) == lowercase(ext) ||
            error("wrong file extension: $path (expected $ext)")

        hdrpath = path
        recpath = load_data ? (
            isfile(path1 * ".REC") ? path1 * ".REC" :
            isfile(path1 * ".rec") ? path1 * ".rec" :
            error(".REC file $path1 does not exist")
        ) : ""

    elseif lowercase(ext1) == ".rec"
        recpath = path
        hdrpath = ext == "" ? (
            isfile(path1 * ".PAR") ? path1 * ".PAR" :
            isfile(path1 * ".XML") ? path1 * ".XML" :
            isfile(path1 * ".par") ? path1 * ".par" :
            isfile(path1 * ".xml") ? path1 * ".xml" :
            error(".PAR/.XML file $path1 does not exist")

        ) : lowercase(ext) == ".par" ? (
            isfile(path1 * ".PAR") ? path1 * ".PAR" :
            isfile(path1 * ".par") ? path1 * ".par" :
            error(".PAR file $path1 does not exist")

        ) : lowercase(ext) == ".xml" ? (
            isfile(path1 * ".XML") ? path1 * ".XML" :
            isfile(path1 * ".xml") ? path1 * ".xml" :
            error(".XML file $path1 does not exist")

        ) : error("file extension $ext is not a valid Philips Rec extension")

    else
        error("file $path does not have valid Philips Rec extension")
    end

    return hdrpath, recpath
end

####
#### .REC
####

readrec(path::AbstractString, hdr::RecHdr) =
    readrec(Float64, path, hdr)

function readrec(T::DataType, path::AbstractString, hdr::RecHdr)
    validate(hdr)
    hdr = hdr.images

    prec = Dict{Int,DataType}(
        8  => UInt8,
        16 => UInt16,
        32 => Float32,
        64 => Float64,
    )

    nx       = Int(hdr[1].Resolution_X)
    ny       = Int(hdr[2].Resolution_X)
    nz       = maximum((i -> i.Key.Slice).(hdr))
    echoes   = unique((i -> i.Key.Echo).(hdr))
    dynamics = unique((i -> i.Key.Dynamic).(hdr))
    phases   = unique((i -> i.Key.Phase).(hdr))
    bvals    = unique((i -> i.Key.BValue).(hdr))
    gorients = unique((i -> i.Key.Grad_Orient).(hdr))
    types    = Dict(t => i for (i, t) in enumerate(unique((i -> i.Key.Type).(hdr))))
    labels   = Dict(l => i for (i, l) in enumerate(unique((i -> i.Key.Label_Type).(hdr))))
    seqs     = Dict(s => i for (i, s) in enumerate(unique((i -> i.Key.Sequence).(hdr))))

    if length(unique((i -> i.Pixel_Size).(hdr))) == 1
        sz = (nx * ny * hdr[1].Pixel_Size÷8 * length(hdr)) - filesize(path)
        sz < 0 && @warn "REC file larger than expected from hdr"
        sz > 0 && @warn "REC file smaller than expected from hdr"
    end

    data = zeros(T,
        nx, ny, nz, length.((echoes, dynamics, phases, bvals, gorients, labels, types, seqs))...
    )

    open(path, "r") do io
        for (i, img) in enumerate(hdr)
            ps = Int(img.Pixel_Size)

            seek(io, img.Key.Index * (nx * ny * ps÷8))
            slice = read!(io, Array{prec[ps]}(undef, nx, ny))

            ss  = inv(img.Scale_Slope)
            ri  = ss * img.Rescale_Intercept / img.Rescale_Slope

            i3  = img.Key.Slice
            i4  = img.Key.Echo
            i5  = img.Key.Dynamic
            i6  = img.Key.Phase
            i7  = img.Key.BValue
            i8  = img.Key.Grad_Orient
            i9  = labels[img.Key.Label_Type]
            i10 = types[img.Key.Type]
            i11 = seqs[img.Key.Sequence]

            @. data[:,:,i3,i4,i5,i6,i7,i8,i9,i10,i11] = ss*slice + ri
        end
    end

    return squeeze(data)
end

####
#### .XML
####

const XMLREC_HEADER           = "PRIDE_V5"
const XMLREC_SERIES_HEADER    = "Series_Info"
const XMLREC_IMAGE_ARR_HEADER = "Image_Array"
const XMLREC_IMAGE_HEADER     = "Image_Info"
const XMLREC_IMAGE_KEY_HEADER = "Key"
const XMLREC_ATTRIB_HEADER    = "Attribute"

function readxml(path::AbstractString)
    doc = parse_file(path)

    xroot = root(doc)
    name(xroot) == XMLREC_HEADER || @warn "unknown XML/REC header: $(name(xroot))"

    # Series Info
    cur = find_element(xroot, XMLREC_SERIES_HEADER)
    cur !== nothing || error("<Series_Info> tag not found")

    Ts = NamedTuple{(fieldnames(SeriesInfo)...,)}(SeriesInfo.types)
    series = SeriesInfo()

    for e in cur[XMLREC_ATTRIB_HEADER]
        key, val = _tryparse(Ts, e)
        val !== nothing ?
            setfield!(series, key, val) :
            @warn "unknown series attribute: $key. Skipping..."
    end

    # Image Info
    cur = find_element(xroot, XMLREC_IMAGE_ARR_HEADER)
    cur !== nothing || error("<Image_Array> tag not found")

    Tk = NamedTuple{(fieldnames(ImageKey)...,)}(ImageKey.types)
    Ti = NamedTuple{(fieldnames(ImageInfo)...,)}(ImageInfo.types)
    images = Vector{ImageInfo}()

    for cur in cur[XMLREC_IMAGE_HEADER]
        k = find_element(cur, XMLREC_IMAGE_KEY_HEADER)
        k !== nothing || error("<Key> tag not found")

        imagekey = ImageKey()
        for e in k[XMLREC_ATTRIB_HEADER]
            key, val = _tryparse(Tk, e)
            val !== nothing ?
                setfield!(imagekey, key, val) :
                @warn "unknown key attribute: $key. Skipping..."
        end

        image = ImageInfo(Key = imagekey)
        for e in cur[XMLREC_ATTRIB_HEADER]
            key, val = _tryparse(Ti, e)
            val !== nothing ?
                setfield!(image, key, val) :
                @warn "unknown image attribute: $key. Skipping..."
        end

        push!(images, image)
    end

    free(doc)

    return RecHdr(series, images)
end

####
#### .PAR
####

const PARSERIES = Dict{Symbol,String}(
    :Patient_Name              => "Patient name",
    :Examination_Name          => "Examination name",
    :Protocol_Name             => "Protocol name",
    :Examination_Date          => "Examination date/time",
    :Examination_Time          => "Examination date/time",
    :Series_Data_Type          => "Series Type",
    :Aquisition_Number         => "Acquisition nr",
    :Reconstruction_Number     => "Reconstruction nr",
    :Scan_Duration             => "Scan Duration",
    :Max_No_Phases             => "Max. number of cardiac phases",
    :Max_No_Echoes             => "Max. number of echoes",
    :Max_No_Slices             => "Max. number of slices",
    :Max_No_Dynamics           => "Max. number of dynamics",
    :Max_No_Mixes              => "Max. number of mixes",
    :Max_No_B_Values           => "Max. number of diffusion",
    :Max_No_Gradient_Orients   => "Max. number of gradient",
    :No_Label_Types            => "Number of label types",
    :Patient_Position          => "Patient position",
    :Preparation_Direction     => "Preparation direction",
    :Technique                 => "Technique",
    :Scan_Resolution_X         => "Scan resolution",
    :Scan_Resolution_Y         => "Scan resolution",
    :Scan_Mode                 => "Scan mode",
    :Repetition_Times          => "Repetition time",
    :FOV_AP                    => "FOV",
    :FOV_FH                    => "FOV",
    :FOV_RL                    => "FOV",
    :Water_Fat_Shift           => "Water Fat",
    :Angulation_AP             => "Angulation",
    :Angulation_FH             => "Angulation",
    :Angulation_RL             => "Angulation",
    :Off_Center_AP             => "Off Centre",
    :Off_Center_FH             => "Off Centre",
    :Off_Center_RL             => "Off Centre",
    :Flow_Compensation         => "Flow",
    :Presaturation             => "Presaturation",
    :Phase_Encoding_Velocity   => "Phase encoding",
    :MTC                       => "MTC",
    :SPIR                      => "SPIR",
    :EPI_factor                => "EPI factor",
    :Dynamic_Scan              => "Dynamic scan",
    :Diffusion                 => "Diffusion   ",
    :Diffusion_Echo_Time       => "Diffusion echo time",
)

const PARKEY = Dict{Symbol,Int}(
    :Slice       => 1,
    :Echo        => 2,
    :Dynamic     => 3,
    :Phase       => 4,
    :BValue      => 42,
    :Grad_Orient => 43,
    :Label_Type  => 49,
    :Type        => 5,
    :Sequence    => 6,
    :Index       => 7,
)

const PARIMAGE = Dict{Symbol,Int}(
    :Pixel_Size                 => 8,
    :Scan_Percentage            => 9,
    :Resolution_X               => 10,
    :Resolution_Y               => 11,
    :Rescale_Intercept          => 12,
    :Rescale_Slope              => 13,
    :Scale_Slope                => 14,
    :Window_Center              => 15,
    :Window_Width               => 16,
    :Slice_Thickness            => 23,
    :Slice_Gap                  => 24,
    :Display_Orientation        => 25,
    :fMRI_Status_Indication     => 27,
    :Image_Type_Ed_Es           => 28,
    :Pixel_Spacing              => 29,  # [29, 30]
    :Echo_Time                  => 31,
    :Dyn_Scan_Begin_Time        => 32,
    :Trigger_Time               => 33,
    :Diffusion_B_Factor         => 34,
    :No_Averages                => 35,
    :Image_Flip_Angle           => 36,
    :Cardiac_Frequency          => 37,
    :Min_RR_Interval            => 38,
    :Max_RR_Interval            => 39,
    :TURBO_Factor               => 40,
    :Inversion_Delay            => 41,
    :Contrast_Type              => 44,
    :Diffusion_Anisotropy_Type  => 45,
    :Diffusion_AP               => 46,
    :Diffusion_FH               => 47,
    :Diffusion_RL               => 48,
    :Angulation_AP              => 17,
    :Angulation_FH              => 18,
    :Angulation_RL              => 19,
    :Offcenter_AP               => 20,
    :Offcenter_FH               => 21,
    :Offcenter_RL               => 22,
    :Slice_Orientation          => 26,
)

function readpar(path::AbstractString)
    doc = read(path, String)

    # Series Info / General Information
    Ts = NamedTuple{(fieldnames(SeriesInfo)...,)}(SeriesInfo.types)
    series = SeriesInfo()

    # lines starting with `.`
    buf = [strip.(split(x.match, ": ")) for x in eachmatch(r"(?<=\n\.).*?\n", doc)];

    for (k, v) in PARSERIES
        str = buf[findfirst(a -> occursin(v, a[1]), buf)][2]
        val =
            k == :Examination_Date ?
                strip(split(str, "/")[1]) |> string :
            k == :Examination_Time ?
                strip(split(str, "/")[2]) |> string :
            k ∈ [:Scan_Resolution_X, :FOV_AP, :Angulation_AP, :Off_Center_AP] ?
                _parse(Ts[k], split(str)[1]) :
            k ∈ [:Scan_Resolution_Y, :FOV_FH, :Angulation_FH, :Off_Center_FH] ?
                _parse(Ts[k], split(str)[2]) :
            k ∈ [:FOV_RL, :Angulation_RL, :Off_Center_RL] ?
                _parse(Ts[k], split(str)[3]) :
            # else
                _parse(Ts[k], str)

        setfield!(series, k, val)
    end

    # Image Info
    Tk = NamedTuple{(fieldnames(ImageKey)...,)}(ImageKey.types)
    Ti = NamedTuple{(fieldnames(ImageInfo)...,)}(ImageInfo.types)
    images = Vector{ImageInfo}()

    # lines starting with ` ` or digit
    buf = [split(x.match) for x in eachmatch(r"(?<=\n)( |\d+).*?\n", doc)];

    for line in buf
        imagekey = ImageKey()
        for (k, v) in PARKEY
            setfield!(imagekey, k, _parse(Tk[k], line[v]))
        end
        image = ImageInfo(Key = imagekey)
        for (k, v) in PARIMAGE
            k == :Pixel_Spacing ?
                setfield!(image, k, _parse(Ti[k], line[v]*" "*line[v+1])) :
                setfield!(image, k, _parse(Ti[k], line[v]))
        end
        push!(images, image)
    end

    return RecHdr(series, images)
end

####
#### Utilities
####

function validate(hdr::RecHdr)
    series = hdr.series
    images = hdr.images

    keys = (i -> i.Key).(images)
    keymax = (v) -> maximum(getfield.(keys, v))
    keylength = (v) -> length(unique(getfield.(keys, v)))
    imglength = (v) -> length(unique(getfield.(images, v)))

    vals = [
        (series.Max_No_Slices, :Slice, "slices"),
        (series.Max_No_Echoes, :Echo, "echoes"),
        (series.Max_No_Dynamics, :Dynamic, "dynamics"),
        (series.Max_No_Phases, :Phase, "phases"),
        (series.Max_No_B_Values, :BValue, "bvalues"),
        (series.Max_No_Gradient_Orients, :Grad_Orient, "gradient orientations"),
    ]

    # Check whether data size in series info is consistent with image info
    wstr1 = (f, s, i) ->
        "maximum number of $f not matching: $s (series info) vs. $i (image info)"

    for (v1,v2,v3) in vals
        v1 == keymax(v2) || @warn wstr1(v3, v1, keymax(v2))
    end

    series.No_Label_Types + 1 == keylength(:Label_Type) ||
        @warn wstr1("label types", series.No_Label_Types, keylength(:Label_Type))

    # Non-exhaustive image info check
    wstr2 = (f, m, l) ->
        "inconsistent number of $f in image info: $m (maximum) vs. $l (length(unique))"

    for (_, v1, v2) in vals
        keymax(v1) == keylength(v1) ||
            @warn wstr2(v2, keymax(v1), keylength(v1))
    end

    keymax(:Index) + 1 == keylength(:Index) ||
        @warn wstr2("indices", keymax(:Index), keylength(:Index))

    imglength(:Pixel_Size) == 1 ||
        @warn "multiple values for `Pixel Size` found"

    imglength(:Slice_Thickness) == 1 ||
        @warn "multiple values for `Slice Thickness` found"

    imglength(:Slice_Gap) == 1 ||
        @warn "multiple values for `Slice Gap` found"

    length(unique(first.(getfield.(images, :Pixel_Spacing)))) == 1 ||
        @warn "multiple values for `Pixel Spacing` (x) found"

    length(unique(last.(getfield.(images, :Pixel_Spacing)))) == 1 ||
        @warn "multiple values for `Pixel Spacing` (y) found"

    imglength(:Resolution_X) == 1 ||
        error("multiple values for `Resolution X` found")

    imglength(:Resolution_Y) == 1 ||
        error("multiple values for `Resolution Y` found")

    return nothing
end

function squeeze(x::Array{T}) where {T}
    N = count(n -> n > 1, size(x))
    dims = ([d for d in 1:ndims(x) if size(x, d) == 1]...,)
    return dropdims(x, dims = dims) :: Array{T,N}
end

_parse(::Type{String}, val::AbstractString) = string(val)

_parse(::Type{T}, val::AbstractString) where {T<:Tuple} =
    (_parse.(T.types, split(val))...,)

_parse(::Type{Bool}, val::AbstractString) =
    (val == "1" || uppercase(val) == "Y") ? true :
    (val == "0" || uppercase(val) == "N") ? false : _parse(Bool, val)

_parse(type, str; kwargs...) = Base.parse(type, str; kwargs...) # fallback

function _tryparse(T::NamedTuple, e::XMLElement)
    key = attribute(e, "Name") |> x -> replace(strip(x), " " => "_") |> Symbol
    val = haskey(T, key) ? _parse(T[key], strip(content(e))) : nothing
    return key, val
end

_tryparse(type, str; kwargs...) = Base.tryparse(type, str; kwargs...) # fallback

end # module
