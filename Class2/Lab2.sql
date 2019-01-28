select ar.name from invoices i
join invoice_items ii 
	on i.invoiceid = ii.invoiceid 
join tracks t 
	on ii.trackid = t.trackid
join albums al 
	on t.albumid = al.albumid
join artists ar 
	on al.artistid = ar.artistid
where i.billingstate = 'CA'
group by ar.artistid
order by sum(ii.Quantity) desc
limit 3;

