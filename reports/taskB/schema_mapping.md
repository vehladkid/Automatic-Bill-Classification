# Source schema -> canonical Task B schema

## sroie
* `merchant_name` <- company
* `merchant_address` <- address
* `invoice_date` <- date
* `total_amount` <- total
* UNSUPPORTED: (none: only 4 fields annotated)

## cord
* `total_amount` <- total_amount
* `subtotal` <- subtotal
* `tax_total` <- tax_amount
* UNSUPPORTED: menu items (line items, not a Task B field); payment_cash (cash tendered, not payment method)

## receiptbench
* `merchant_name` <- seller_name
* `invoice_number` <- invoice_number
* `invoice_date` <- std_invoice_time
* `total_amount` <- std_total, or orig_total when std_total is the placeholder '0.00' (182 docs); neither usable -> total not scored
* `currency` <- std_curr
* `gstin` <- tax_number (only if GSTIN-format)
* `cgst/sgst/igst` <- detail[] lines whose content contains CGST/SGST(UTGST)/IGST (India only)
* `origin/destination` <- departure/arrival (flight/train/bus types; city part)
* `travel_date` <- std_start_time (flight/train/bus)
* `check_in/check_out` <- std_start_time/std_end_time (hotel)
* UNSUPPORTED: seller_address (city-level 'IN-Bengaluru', not a street address); tax_number when not GSTIN-format (foreign tax ids); std_end_time for non-hotel types; detail line items; tax_total/subtotal (not separable from detail)

## invoices_pt
* `merchant_name` <- company
* `merchant_address` <- address
* `invoice_date` <- date
* `total_amount` <- total
* `invoice_number` <- invoice_number
* `tax_total` <- iva_amount
* UNSUPPORTED: nif_buyer / nif_seller (tax ids; no canonical field)

## synthetic_in
* `merchant_name` <- merchant
* `merchant_address` <- address
* `invoice_number` <- invoice_number
* `invoice_date` <- date
* `subtotal` <- subtotal
* `tax_total` <- tax_amount
* `total_amount` <- total_amount
* `currency` <- currency
* `payment_method` <- payment_method
* `gstin` <- GSTIN
* `cgst/sgst/igst` <- tax_split
* `origin/destination/pnr/train_number/flight_number/vehicle_number` <- origin/destination/pnr/train_no/flight_no/vehicle_no
* UNSUPPORTED: customer_name (role depends on category); phone; round_off; consumer_no/account_no/landlord(_pan)/units/upi_ref (no canonical field)

## synthetic_b
* `*` <- all canonical fields incl. hsn_sac, hotel stay, medical, employee, passenger/ticket/bus (generated with the rendered text)

## Not mapped anywhere
* funsd: form entities (non-expense forms; not a receipt schema)
* all real sources: hsn_sac, employee_*, pnr, ticket_number, passenger_name, flight/train/bus/vehicle number, room_number, number_of_nights, patient/doctor/prescription/pharmacy: NO real annotation exists